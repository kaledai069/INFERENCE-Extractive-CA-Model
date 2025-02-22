import math
import string
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.special import log_softmax, softmax
from tqdm import trange

from Utils_inf import print_grid, get_word_flips
from Solver_inf import Solver
from Models_inf import setup_t5_reranker, t5_reranker_score_with_clue


# the probability of each alphabetical character in the crossword
UNIGRAM_PROBS = [('A', 0.0897379968935765), ('B', 0.02121248877769636), ('C', 0.03482206634145926), ('D', 0.03700942543460491), ('E', 0.1159773210750429), ('F', 0.017257461694024614), ('G', 0.025429024796296124), ('H', 0.033122967601502), ('I', 0.06800036223479956), ('J', 0.00294611331754349), ('K', 0.013860682888259786), ('L', 0.05130800574373874), ('M', 0.027962776827660175), ('N', 0.06631994270448001), ('O', 0.07374646543246745), ('P', 0.026750756212433214), ('Q', 0.001507814175439393), ('R', 0.07080460813737305), ('S', 0.07410988246048224), ('T', 0.07242993582154593), ('U', 0.0289272388037645), ('V', 0.009153522059555467), ('W', 0.01434705167591524), ('X', 0.003096729223103298), ('Y', 0.01749958208224007), ('Z', 0.002659777584995724)]

# the LETTER_SMOOTHING_FACTOR controls how much we interpolate with the unigram LM. TODO this should be tuned. 
# Right now it is set according to the probability that the answer is not in the answer set
LETTER_SMOOTHING_FACTOR = [0.0, 0.0, 0.04395604395604396, 0.0001372495196266813, 0.0005752186417796561, 0.0019841824329989103, 0.0048042463338563764, 0.013325257419745608, 0.027154447774285505, 0.06513517299341645, 0.12527790128946198, 0.22003002358996354, 0.23172376584839494, 0.254873006497342, 0.3985086992543496, 0.2764976958525346, 0.672645739910314, 0.6818181818181818, 0.8571428571428571, 0.8245614035087719, 0.8, 0.71900826446281, 0.0]

class BPVar:
    def __init__(self, name, variable, candidates, cells):
        self.name = name # key from crossword.variables i.e. 1A, 2D, 3A
        cells_by_position = {}
        for cell in cells: # every cells or letter box that a particular variable or filling takes into consideration
            cells_by_position[cell.position] = cell # cell.position (0,0) -> cell -> BPCell
            cell._connect(self)
        self.length = len(cells) # obviously the length of the answer
        self.ordered_cells = [cells_by_position[pos] for pos in variable['cells']]
        self.candidates = candidates
        self.words = self.candidates['words']
        self.word_indices = np.array([[string.ascii_uppercase.index(l) for l in fill] for fill in self.candidates['words']]) # words x length of letter indices
        self.scores = -np.array([self.candidates['weights'][fill] for fill in self.candidates['words']]) # the incoming 'weights' are costs
        self.prior_log_probs = log_softmax(self.scores)
        self.log_probs = log_softmax(self.scores)
        self.directional_scores = [np.zeros(len(self.log_probs)) for _ in range(len(self.ordered_cells))]
    
    def _propagate_to_var(self, other, belief_state):
        assert other in self.ordered_cells
        other_idx = self.ordered_cells.index(other)
        letter_scores = belief_state
        self.directional_scores[other_idx] = letter_scores[self.word_indices[:, other_idx]]
    
    def _postprocess(self, all_letter_probs):
        # unigram smoothing
        unigram_probs = np.array([x[1] for x in UNIGRAM_PROBS])
        for i in range(len(all_letter_probs)):
            all_letter_probs[i] = (1 - LETTER_SMOOTHING_FACTOR[self.length]) * all_letter_probs[i] + LETTER_SMOOTHING_FACTOR[self.length] * unigram_probs
        return all_letter_probs
    
    def sync_state(self):
        self.log_probs = log_softmax(sum(self.directional_scores) + self.prior_log_probs)
    
    def propagate(self):
        all_letter_probs = []
        for i in range(len(self.ordered_cells)):
            word_scores = self.log_probs - self.directional_scores[i]
            word_probs = softmax(word_scores)
            letter_probs = (self.candidates['bit_array'][:, i] * np.expand_dims(word_probs, axis=0)).sum(axis=1) + 1e-8
            all_letter_probs.append(letter_probs)
        all_letter_probs = self._postprocess(all_letter_probs) # unigram postprocessing
        for i, cell in enumerate(self.ordered_cells):
            cell._propagate_to_cell(self, np.log(all_letter_probs[i]))


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


class BPSolver(Solver):
    def __init__(self, 
                 crossword, 
                 model_path,
                 ans_tsv_path,
                 dense_embd_path,
                 reranker_path, 
                 max_candidates = 100,
                 process_id = 0,
                 model_type = 'bert',
                 **kwargs):
        super().__init__(crossword, 
                         model_path,
                         ans_tsv_path,
                         dense_embd_path,
                         max_candidates = max_candidates,
                         process_id  = process_id,
                         model_type = model_type,
                         **kwargs)
        self.crossword = crossword
        self.reranker_path = reranker_path
        self.reranker_model_type = 't5-small'

        # our answer set
        self.answer_set = set()
        with open(ans_tsv_path, 'r') as rf: 
            for line in rf:
                w = ''.join([c.upper() for c in (line.split('\t')[-1]).upper() if c in string.ascii_uppercase])
                self.answer_set.add(w)
        self.reset()
    
    def reset(self):
        self.bp_cells = []
        self.bp_cells_by_clue = defaultdict(lambda: [])
        for position, clue_pair in self.crossword.grid_cells.items():
            cell = BPCell(position, clue_pair)
            self.bp_cells.append(cell)
            for clue in clue_pair:
                self.bp_cells_by_clue[clue].append(cell)
        self.bp_vars = []
        for key, value in self.crossword.variables.items():
            # if key == '1A':
            #     print('-'*100)
            #     print(self.candidates[key]['words'])
            #     print(self.candidates[key]['bit_array'].shape)
            #     print(self.candidates[key]['weights'])
            #     print('-'*100)
            var = BPVar(key, value, self.candidates[key], self.bp_cells_by_clue[key])
            # print('*'*100)
            # print(self.bp_cells_by_clue[key])
            # print('*'*100)
            self.bp_vars.append(var)

    def extract_float(self, input_string):
        pattern = r"\d+\.\d+"
        matches = re.findall(pattern, input_string)
        float_numbers = [float(match) for match in matches]
        return float_numbers
    
    def solve(self, num_iters=10, iterative_improvement_steps=5, return_greedy_states = False, return_ii_states = False):
        # run solving for num_iters iterations
        print('\nBeginning Belief Propagation Iteration Steps: ')
        for _ in trange(num_iters):
            for var in self.bp_vars:
                var.propagate()
            for cell in self.bp_cells:
                cell.sync_state()
            for cell in self.bp_cells:
                cell.propagate()
            for var in self.bp_vars:
                var.sync_state()
        print('Belief Propagation Iteration Complete\n')
       
        # Get the current based grid based on greedy selection from the marginals
        if return_greedy_states:
            grid, all_grids = self.greedy_sequential_word_solution(return_grids = True)
        else:
            grid = self.greedy_sequential_word_solution()
            all_grids = []
        
        # properly save all the outputs results:
        output_results = {}
        output_results['first pass model'] = {}
        output_results['first pass model']['grid'] = grid

        # save first pass model grid, and letter accuracies
        _, accu_log = self.evaluate(grid, False)
        [ori_letter_accu, ori_word_accu] = self.extract_float(accu_log)
        output_results['first pass model']['letter accuracy'] = ori_letter_accu
        output_results['first pass model']['word accuracy'] = ori_word_accu

        if iterative_improvement_steps < 1 or ori_letter_accu == 100.0:
            if return_greedy_states or return_ii_states:
                return output_results, all_grids
            else:
                return output_results, self.bp_cells, self.bp_cells_by_clue
        
        '''
            Iterative Improvement with t5-small starts from here.
        '''
        self.reranker, self.tokenizer = setup_t5_reranker(self.reranker_path, self.reranker_model_type)
        
        output_results['second pass model'] = {}
        output_results['second pass model']['all grids'] = []
        output_results['second pass model']['all letter accuracy'] = []
        output_results['second pass model']['all word accuracy'] = []

        for i in range(iterative_improvement_steps):
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

            if return_ii_states:
                all_grids.append(deepcopy(grid))

        temp_lett_accu_list = output_results['second pass model']['all letter accuracy'].copy()
        ii_max_index = temp_lett_accu_list.index(max(temp_lett_accu_list))

        output_results['second pass model']['final grid'] = output_results['second pass model']['all grids'][ii_max_index]
        output_results['second pass model']['final letter'] = output_results['second pass model']['all letter accuracy'][ii_max_index]
        output_results['second pass model']['final word'] = output_results['second pass model']['all word accuracy'][ii_max_index]

        if return_greedy_states or return_ii_states:
            return output_results, all_grids
        else:
            return output_results

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

    def greedy_sequential_word_solution(self, return_grids = False):
        all_grids = []
        # after we've run BP, we run a simple greedy search to get the final. 
        # We repeatedly pick the highest-log-prob candidate across all clues which fits the grid, and fill it. 
        # at the end, if you have any cells left (due to missing gold candidates) just fill it with the argmax on that letter.
        cache = [(deepcopy(var.words), deepcopy(var.log_probs)) for var in self.bp_vars]

        grid = [["" for _ in row] for row in self.crossword.letter_grid]
        unfilled_cells = set([cell.position for cell in self.bp_cells])
        for var in self.bp_vars:
            # postprocess log probs to estimate probability that you don't have the right candidate
            var.log_probs = var.log_probs + math.log(1 - LETTER_SMOOTHING_FACTOR[var.length])
        best_per_var = [var.log_probs.max() for var in self.bp_vars]
        while not all([x is None for x in best_per_var]):
            all_grids.append(deepcopy(grid))
            best_index = best_per_var.index(max([x for x in best_per_var if x is not None]))
            best_var = self.bp_vars[best_index]
            best_word = best_var.words[best_var.log_probs.argmax()]
            for i, cell in enumerate(best_var.ordered_cells):
                letter = best_word[i]
                grid[cell.position[0]][cell.position[1]] = letter
                if cell.position in unfilled_cells:
                    unfilled_cells.remove(cell.position)
                for var in cell.crossing_vars:
                    if var != best_var:
                        cell_index = var.ordered_cells.index(cell)
                        keep_indices = [j for j in range(len(var.words)) if var.words[j][cell_index] == letter]
                        var.words = [var.words[j] for j in keep_indices]
                        var.log_probs = var.log_probs[keep_indices]
                        var_index = self.bp_vars.index(var)
                        if len(keep_indices) > 0:
                            best_per_var[var_index] = var.log_probs.max()
                        else:
                            best_per_var[var_index] = None
            best_var.words = []
            best_var.log_probs = best_var.log_probs[[]]
            best_per_var[best_index] = None

        unfilled_cells_count = 0
        for cell in self.bp_cells:
            if cell.position in unfilled_cells:
                unfilled_cells_count += 1
                grid[cell.position[0]][cell.position[1]] = string.ascii_uppercase[cell.log_probs.argmax()]
                
        for var, (words, log_probs) in zip(self.bp_vars, cache): # restore state
            var.words = words
            var.log_probs = log_probs
        if return_grids:
            return grid, all_grids
        else:
            return grid
        
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
            #     print('original:', original_fill, 'modified:', modified_fill)
            #     print('gold answer', self.crossword.variables[clue_index]['gold'])
            #     print('clue', self.crossword.variables[clue_index]['clue'])
            # print('original score:', original_grid_score, 'modified score:', modified_grid_score)
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