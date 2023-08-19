import math
import string
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.special import log_softmax, softmax
from tqdm import trange

from Utils_inf import print_grid, get_word_flips
from Solver_inf import Solver

# the probability of each alphabetical character in the crossword
UNIGRAM_PROBS = [('A', 0.0897379968935765), ('B', 0.02121248877769636), ('C', 0.03482206634145926), ('D', 0.03700942543460491), ('E', 0.1159773210750429), ('F', 0.017257461694024614), ('G', 0.025429024796296124), ('H', 0.033122967601502), ('I', 0.06800036223479956), ('J', 0.00294611331754349), ('K', 0.013860682888259786), ('L', 0.05130800574373874), ('M', 0.027962776827660175), ('N', 0.06631994270448001), ('O', 0.07374646543246745), ('P', 0.026750756212433214), ('Q', 0.001507814175439393), ('R', 0.07080460813737305), ('S', 0.07410988246048224), ('T', 0.07242993582154593), ('U', 0.0289272388037645), ('V', 0.009153522059555467), ('W', 0.01434705167591524), ('X', 0.003096729223103298), ('Y', 0.01749958208224007), ('Z', 0.002659777584995724)]

# the LETTER_SMOOTHING_FACTOR controls how much we interpolate with the unigram LM. TODO this should be tuned. 
# Right now it is set according to the probability that the answer is not in the answer set
LETTER_SMOOTHING_FACTOR = [0.0, 0.0, 0.04395604395604396, 0.0001372495196266813, 0.0005752186417796561, 0.0019841824329989103, 0.0048042463338563764, 0.013325257419745608, 0.027154447774285505, 0.06513517299341645, 0.12527790128946198, 0.22003002358996354, 0.23172376584839494, 0.254873006497342, 0.3985086992543496, 0.2764976958525346, 0.672645739910314, 0.6818181818181818, 0.8571428571428571, 0.8245614035087719, 0.8, 0.71900826446281, 0.0]

class BPVar:
    def __init__(self, name, variable, candidates, cells):
        self.name = name
        cells_by_position = {}
        for cell in cells:
            cells_by_position[cell.position] = cell
            cell._connect(self)
        self.length = len(cells)
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
        assert len(self.crossing_vars) == 2
        for i, v in enumerate(self.crossing_vars):
            v._propagate_to_var(self, self.directional_scores[1-i])


class BPSolver(Solver):
    def __init__(self, 
                 crossword, 
                 model_path,
                 ans_tsv_path,
                 dense_embd_path,
                 max_candidates = 500000,
                 process_id=0,
                 **kwargs):
        super().__init__(crossword, 
                         model_path,
                         ans_tsv_path,
                         dense_embd_path,
                         max_candidates=max_candidates,
                         process_id=process_id,
                         **kwargs)
        self.crossword = crossword

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
            var = BPVar(key, value, self.candidates[key], self.bp_cells_by_clue[key])
            self.bp_vars.append(var)
    
    def solve(self, num_iters=10, iterative_improvement_steps=5, return_greedy_states = False, return_ii_states = False):
        # run solving for num_iters iterations
        print('beginning BP iterations')
        for _ in trange(num_iters):
            for var in self.bp_vars:
                var.propagate()
            for cell in self.bp_cells:
                cell.sync_state()
            for cell in self.bp_cells:
                cell.propagate()
            for var in self.bp_vars:
                var.sync_state()
        print('done BP iterations')
       
        # Get the current based grid based on greedy selection from the marginals
        if return_greedy_states:
            grid, all_grids = self.greedy_sequential_word_solution(return_grids = True)
        else:
            grid = self.greedy_sequential_word_solution()
            all_grids = []
        grid = self.greedy_sequential_word_solution()
        # print('=====Greedy search grid=====')
        # print_grid(grid)

        if iterative_improvement_steps < 1:
            if return_greedy_states or return_ii_states:
                return grid, all_grids
            else:
                return grid
    
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
            # print('greedy filling in', best_word)
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
        for cell in self.bp_cells:
            if cell.position in unfilled_cells:
                grid[cell.position[0]][cell.position[1]] = string.ascii_uppercase[cell.log_probs.argmax()]
        
        for var, (words, log_probs) in zip(self.bp_vars, cache): # restore state
            var.words = words
            var.log_probs = log_probs
        if return_grids:
            return grid, all_grids
        else:
            return grid