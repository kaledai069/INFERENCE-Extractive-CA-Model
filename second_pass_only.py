import re
import string 
import numpy as np

from copy import deepcopy
from scipy.special import log_softmax, softmax
from Crossword_inf import Crossword
from Utils_inf import print_grid, get_word_flips
from Models_inf import setup_t5_reranker, t5_reranker_score_with_clue

class BPCell:
    def __init__(self, position, clue_pair):
        self.crossing_clues = clue_pair
        self.position = tuple(position)
        self.letters = list(string.ascii_uppercase)
        self.log_probs = np.log(np.array([1./len(self.letters) for _ in range(len(self.letters))]))
        self.crossing_vars = []
        self.directional_scores = []
        self.prediction = {}
    
    def _connect(self, other):
        self.crossing_vars.append(other)
        self.directional_scores.append(None)
        assert len(self.crossing_vars) <= 2

    def _propagate_to_cell(self, other, belief_state):
        assert other in self.crossing_vars
        other_idx = self.crossing_vars.index(other)
        self.directional_scores[other_idx] = belief_state
    
    def sync_state(self):
        self.log_probs = log_softmax(sum(self.directional_scores))

    def propagate(self):
        # assert len(self.crossing_vars) == 2
        try:
            for i, v in enumerate(self.crossing_vars):
                v._propagate_to_var(self, self.directional_scores[1-i])
        except IndexError:
            pass

class SecondPassSolver():

    def __init__(self, 
                 first_pass_grid,
                 crossword, 
                 ans_tsv_path,
                 reranker_path, 
                 reranker_model_type,
                 bp_cells,
                 bp_cells_by_clue,
                 iterative_improvement_steps = 3):
        
        self.crossword = crossword
        self.reranker_path = reranker_path
        self.reranker_model_type = reranker_model_type
        self.iterative_improvement_steps = iterative_improvement_steps
        self.first_pass_grid = first_pass_grid
        self.bp_cells = bp_cells
        self.bp_cells_by_clue = bp_cells_by_clue

        # our answer set
        self.answer_set = set()
        with open(ans_tsv_path, 'r') as rf: 
            for line in rf:
                w = ''.join([c.upper() for c in (line.split('\t')[-1]).upper() if c in string.ascii_uppercase])
                self.answer_set.add(w)

    def solve(self):
        output_results = {}
        output_results['second pass model'] = {}
        output_results['second pass model']['all grids'] = []
        output_results['second pass model']['all letter accuracy'] = []
        output_results['second pass model']['all word accuracy'] = []

        grid = self.first_pass_grid

        self.reranker, self.tokenizer = setup_t5_reranker(self.reranker_path, self.reranker_model_type)

        for i in range(self.iterative_improvement_steps):

            grid, did_iterative_improvement_make_edit = self.iterative_improvement(grid)

            _, accu_log = self.evaluate(grid, False)
            [temp_letter_accu, temp_word_accu] = self.extract_float(accu_log)
            print(f"{i+1}th iteration: {accu_log}")

            # save grid & accuracies at each iteration
            output_results['second pass model']['all grids'].append(grid)
            output_results['second pass model']['all letter accuracy'].append(temp_letter_accu)
            output_results['second pass model']['all word accuracy'].append(temp_word_accu)

            if not did_iterative_improvement_make_edit or temp_letter_accu == 100.0:
                break

        temp_lett_accu_list = output_results['second pass model']['all letter accuracy'].copy()
        ii_max_index = temp_lett_accu_list.index(max(temp_lett_accu_list))

        output_results['second pass model']['final grid'] = output_results['second pass model']['all grids'][ii_max_index]
        output_results['second pass model']['final letter'] = output_results['second pass model']['all letter accuracy'][ii_max_index]
        output_results['second pass model']['final word'] = output_results['second pass model']['all word accuracy'][ii_max_index]

        return output_results

    def extract_float(self, input_string):
        pattern = r"\d+\.\d+"
        matches = re.findall(pattern, input_string)
        float_numbers = [float(match) for match in matches]
        return float_numbers

    def evaluate(self, solution, print_log = True):
        letters_correct = 0
        letters_total = 0
        for i in range(len(self.crossword.letter_grid)):
            for j in range(len(self.crossword.letter_grid[0])):
                if self.crossword.letter_grid[i][j] != "":
                    letters_correct += (self.crossword.letter_grid[i][j] == solution[i][j])
                    letters_total += 1
        words_correct = 0
        words_total = 0
        for var in self.crossword.variables:
            cells = self.crossword.variables[var]["cells"]
            matching_cells = [self.crossword.letter_grid[cell[0]][cell[1]] == solution[cell[0]][cell[1]] for cell in cells]
            if len(cells) == sum(matching_cells):
                words_correct += 1
            words_total += 1

        letter_frac_log = "Letters Correct: {}/{} | Words Correct: {}/{}".format(int(letters_correct), int(letters_total), int(words_correct), int(words_total))
        letter_acc_log = "Letters Correct: {}% | Words Correct: {}%".format(float(letters_correct/letters_total*100), float(words_correct/words_total*100))

        if print_log:
            print(letter_frac_log)
            print(letter_acc_log)
        
        return letter_frac_log, letter_acc_log

    
    def get_candidate_replacements(self, uncertain_answers, grid):
        # find alternate answers for all the uncertain words
        candidate_replacements = []
        replacement_id_set = set()

        # check against dictionaries
        for clue in uncertain_answers.keys():
            initial_word = uncertain_answers[clue]
            clue_flips = get_word_flips(initial_word, 10) # flip then segment
            clue_positions = [key for key, value in self.crossword.variables.items() if value['clue'] == clue]
            for clue_position in clue_positions:
                cells = sorted([cell for cell in self.bp_cells if clue_position in cell.crossing_clues], key=lambda c: c.position)
                if len(cells) == len(initial_word):
                    break
            for flip in clue_flips:
                if len(flip) != len(cells):
                    import pdb; pdb.set_trace()
                assert len(flip) == len(cells)
                for i in range(len(flip)):
                    if flip[i] != initial_word[i]:
                        candidate_replacements.append([(cells[i], flip[i])])
                        break

        # also add candidates based on uncertainties in the letters, e.g., if we said P but G also had some probability, try G too
        for cell_id, cell in enumerate(self.bp_cells): 
            probs = np.exp(cell.log_probs)
            above_threshold = list(probs > 0.01)
            new_characters = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in range(26) if above_threshold[i]]
            # used = set()
            # new_characters = [x for x in new_characters if x not in used and (used.add(x) or True)] # unique the set
            new_characters = [x for x in new_characters if x != grid[cell.position[0]][cell.position[1]]] # ignore if its the same as the original solution
            if len(new_characters) > 0: 
                for new_character in new_characters:
                    id = '_'.join([str(cell.position), new_character])
                    if id not in replacement_id_set:
                        candidate_replacements.append([(cell, new_character)])
                    replacement_id_set.add(id)

        # create composite flips based on things in the same row/column
        composite_replacements = []
        for i in range(len(candidate_replacements)):
            for j in range(i+1, len(candidate_replacements)):
                flip1, flip2 = candidate_replacements[i], candidate_replacements[j]
                if flip1[0][0] != flip2[0][0]:
                    if len(set(flip1[0][0].crossing_clues + flip2[0][0].crossing_clues)) < 4: # shared clue
                        composite_replacements.append(flip1 + flip2)

        candidate_replacements += composite_replacements

        #print('\ncandidate replacements')
        for cr in candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in cr:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            variables = set(sum([cell.crossing_vars for cell, _ in cr], []))
            for var in variables:
                original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                modified_fill = ''.join([modified_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                #print('original:', original_fill, 'modified:', modified_fill)
        
        return candidate_replacements

    def get_uncertain_answers(self, grid):
        original_qa_pairs = {} # the original puzzle preds that we will try to improve
        # first save what the argmax word-level prediction was for each grid cell just to make life easier
        for var in self.crossword.variables:
            # read the current word off the grid  
            cells = self.crossword.variables[var]["cells"]
            word = []
            for cell in cells:
                word.append(grid[cell[0]][cell[1]])
            word = ''.join(word)
            for cell in self.bp_cells: # loop through all cells
                if cell.position in cells: # if this cell is in the word we are currently handling
                    # save {clue, answer} pair into this cell
                    cell.prediction[self.crossword.variables[var]['clue']] = word
                    original_qa_pairs[self.crossword.variables[var]['clue']] = word

        uncertain_answers = {}

        # find uncertain answers
        # right now the heuristic we use is any answer that is not in the answer set
        for clue in original_qa_pairs.keys():
            if original_qa_pairs[clue] not in self.answer_set:
                uncertain_answers[clue] = original_qa_pairs[clue]

        return uncertain_answers
    
    def score_grid(self, grid):
        clues = []
        answers = []
        for clue, cells in self.bp_cells_by_clue.items():
            letters = ''.join([grid[cell.position[0]][cell.position[1]] for cell in sorted(list(cells), key=lambda c: c.position)])
            clues.append(self.crossword.variables[clue]['clue'])
            answers.append(letters)
        scores = t5_reranker_score_with_clue(self.reranker, self.tokenizer, self.reranker_model_type, clues, answers)
        return sum(scores)
    
    def iterative_improvement(self, grid):
        # check the grid for uncertain areas and save those words to be analyzed in local search, aka looking for alternate candidates
        uncertain_answers = self.get_uncertain_answers(grid) 
        self.candidate_replacements = self.get_candidate_replacements(uncertain_answers, grid)

        # print('\nstarting iterative improvement')
        original_grid_score = self.score_grid(grid)
        possible_edits = []
        for replacements in self.candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in replacements:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            modified_grid_score = self.score_grid(modified_grid)
            # print('candidate edit')
            variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
            for var in variables:
                original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                modified_fill = ''.join([modified_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                clue_index = list(set(var.ordered_cells[0].crossing_clues).intersection(*[set(cell.crossing_clues) for cell in var.ordered_cells]))[0]

            if modified_grid_score - original_grid_score > 0.5:
                # print('found a possible edit')
                possible_edits.append((modified_grid, modified_grid_score, replacements))
            # print()
        
        if len(possible_edits) > 0:
            variables_modified = set()
            possible_edits = sorted(possible_edits, key=lambda x: x[1], reverse=True)
            selected_edits = []
            for edit in possible_edits:
                replacements = edit[2]
                variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
                if len(variables_modified.intersection(variables)) == 0: # we can do multiple updates at once if they don't share clues
                    variables_modified.update(variables)
                    selected_edits.append(edit)

            new_grid = deepcopy(grid)
            for edit in selected_edits:
                # print('\nactually applying edit')
                replacements = edit[2]
                for cell, letter in replacements:
                    new_grid[cell.position[0]][cell.position[1]] = letter
                variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
                for var in variables:
                    original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                    modified_fill = ''.join([new_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                    # print('original:', original_fill, 'modified:', modified_fill)
            return new_grid, True
        else:
            return grid, False