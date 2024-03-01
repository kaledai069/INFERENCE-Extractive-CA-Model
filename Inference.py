# import pandas as pd
# import re
# import tokenizers
# import puz
# import os
# import numpy as np
# import streamlit as st
# import scipy
# import sys
# import subprocess
# import copy

# from itertools import zip_longest
# from copy import deepcopy
# import regex

# from Models_inf import setup_closedbook, DPRForCrossword
# from Utils_inf import print_grid


import argparse
import time

import requests
import json
import datetime
import json

from copy import deepcopy
from Crossword_inf import Crossword
from BPSolver_inf import BPSolver
from Strict_json import json_CA_json_converter
from Draw_grid import draw_grid
from Normal_utils_inf import puz_to_json, fetch_nyt_crossword
from second_pass_only import SecondPassSolver

MODEL_CONFIG = {
	'bert': 
	{
		'MODEL_PATH' : "./Inference_components/dpr_biencoder_trained_EPOCH_2_COMPLETE.bin",
		'ANS_TSV_PATH': "./Inference_components/all_answer_list.tsv",
		'DENSE_EMBD_PATH': "./Inference_components/embeddings_BERT_EPOCH_2_COMPLETE0.pkl"
	},
	'distilbert': 
	{
		'MODEL_PATH': "./Inference_components/distilbert_EPOCHs_7_COMPLETE.bin", 
		'ANS_TSV_PATH': "./Inference_components/all_answer_list.tsv",
		'DENSE_EMBD_PATH': "./Inference_components/distilbert_7_epochs_embeddings.pkl"
	},
	't5_small':
	{
		'MODEL_PATH': './Inference_components/t5_small_new_dataset_2EPOCHS/'
	}
}

parser = argparse.ArgumentParser(description = "Args for the model types and option to do-second pass")
parser.add_argument(
					'--crossword_path', 
					default = "nothing", 
					type=str, 
					help='Path to crossword JSON file.'
				   )

parser.add_argument(
					'--date', 
					type = str, 
					help = 'Crossdate to inference to.'
				   )

parser.add_argument(
					'--model', 
					type = str, 
					default = "distilbert", 
					help = "Model type to inference with."
				   )

parser.add_argument(
					'--second_pass', 
					type = bool, 
					default = 'False', 
					help = 'Whether to use second pass model or not.'
				   )
parser.add_argument(
					'--crossword_type',
					type = str,
					default = 'date',
					help  = "Set the type of crossword: Options ['date', 'puz', 'json']"
					)
parser.add_argument
args = parser.parse_args()

MODEL_TYPE = vars(args)['model']
DO_SECOND_PASS = vars(args)['second_pass']
CROSSWORD_TYPE = vars(args)['crossword_type']

if CROSSWORD_TYPE == 'date':
	DATE = vars(args)['date']

# Run the model off the 'DATE' -> "20XX/XX/XX" or from the .puz Crossword data file. 
if CROSSWORD_TYPE == 'date':
	puzzle = fetch_nyt_crossword(args.date)
	# puzzle = json_CA_json_converter(puzzle, False)
elif CROSSWORD_TYPE == 'puz':
	puzzle = puz_to_json(args.crossword_path)
elif CROSSWORD_TYPE == 'json':
	puzzle = json_CA_json_converter(args.crossword_path, True)

for dim in ['across', 'down']:
	for grid_num in puzzle['clues'][dim].keys():
		clue_answer_list = puzzle['clues'][dim][grid_num]
		clue_section = clue_answer_list[0]
		ans_section = clue_answer_list[1]
		clue_section = clue_section.replace("&quot;", "").replace("&#39;", "").replace("<em>", "").replace("</em>", "")
		puzzle['clues'][dim][grid_num] = [clue_section, ans_section]

all_clues = puzzle['clues']

rows = puzzle['metadata']['rows']
cols = puzzle['metadata']['cols']

across_clue_data = []
down_clue_data = []

for dim in ['across', 'down']:
	for key in all_clues[dim].keys():
		clue = all_clues[dim][key][0]
		if dim == 'across':
			across_clue_data.append([key, clue])
		else:
			down_clue_data.append([key, clue])

all_clue_info = [across_clue_data, down_clue_data]

# this crossword variable or object should be separately used for second pass only module too 
crossword = Crossword(puzzle)

choosen_model_path = MODEL_CONFIG[MODEL_TYPE]['MODEL_PATH']
ans_list_path = MODEL_CONFIG[MODEL_TYPE]['ANS_TSV_PATH']
dense_embedding_path = MODEL_CONFIG[MODEL_TYPE]['DENSE_EMBD_PATH']

second_pass_model_path = MODEL_CONFIG['t5_small']['MODEL_PATH']

# print(choosen_model_path, ans_list_path, dense_embedding_path)
# try: 
start_time = time.time()
solver = BPSolver(
					crossword, 
					model_path = choosen_model_path, 
					ans_tsv_path = ans_list_path, 
					dense_embd_path = dense_embedding_path,
					reranker_path = second_pass_model_path, 
					max_candidates = 40000, 
					model_type = MODEL_TYPE
				)


output, bp_cells, bp_cells_by_clue = solver.solve(num_iters = 60, iterative_improvement_steps = 0)
first_pass_grid = output['first pass model']['grid']

second_pass_solver = SecondPassSolver(
										first_pass_grid = first_pass_grid,
										crossword = crossword,
										ans_tsv_path = ans_list_path, 
										reranker_path =  second_pass_model_path,
										reranker_model_type = 't5-small',
										bp_cells = bp_cells,
										bp_cells_by_clue = bp_cells_by_clue,
										iterative_improvement_steps = 2
									 )
second_pass_output = second_pass_solver.solve()
print(second_pass_output)

end_time = time.time()
# except: 
# 	print("Error Occured for date: ", args.date)
# 	accu_list = []

print("Total Inference Time: ", end_time - start_time, " seconds")

# solution = [['L', 'A', 'M', 'O', 'R', '', 'S', 'M', 'A', 'C', 'K', '', 'U', 'S', 'A'], 
# 	    	['E', 'L', 'U', 'D', 'E', '', 'P', 'A', 'L', 'E', 'O', '', 'N', 'A', 'B'], 
# 			['W', 'I', 'G', 'D', 'S', 'E', 'C', 'T', 'I', 'O', 'N', '', 'H', 'U', 'E'], 
# 			['D', 'I', 'S', 'S', 'I', 'P', 'A', 'T', 'E', '', 'I', 'D', 'Y', 'L', 'L'], 
# 			['', '', '', 'A', 'D', 'O', '', '', '', 'E', 'C', 'I', 'G', '', ''], 
# 			['', 'B', 'A', 'R', 'E', 'X', 'A', 'M', 'I', 'N', 'A', 'T', 'I', 'O', 'N'], 
# 			['L', 'A', 'C', 'E', '', 'Y', 'I', 'E', 'L', 'D', '', 'Z', 'E', 'R', 'E'], 
# 			['A', 'T', 'E', '', '', '', 'D', 'D', 'E', '', '', '', 'N', 'E', 'T'], 
# 			['I', 'H', 'O', 'P', '', 'R', 'A', 'I', 'N', 'S', '', 'R', 'I', 'C', 'E'], 
# 			['R', 'E', 'F', 'E', 'L', 'X', 'N', 'C', 'E', 'C', 'H', 'E', 'C', 'K', ''], 
# 			['', '', 'C', 'R', 'U', 'X', '', '', '', 'R', 'U', 'M', '', '', ''], 
# 			['H', 'E', 'L', 'E', 'N', '', 'P', 'O', 'K', 'E', 'B', 'O', 'W', 'L', 'S'], 
# 			['E', 'M', 'U', '', 'D', 'O', 'U', 'B', 'L', 'E', 'C', 'R', 'O', 'S', 'S'], 
# 			['R', 'I', 'B', '', 'M', 'A', 'R', 'I', 'E', '', 'A', 'S', 'W', 'A', 'N'], 
# 			['E', 'T', 'S', '', 'C', 'R', 'E', 'T', 'E', '', 'P', 'E', 'S', 'T', 'S']]

print(output)

first_pass_output = output['first pass model']
first_pass_solution = deepcopy(output['first pass model']['grid']) # gives the grid
first_pass_accu_list = [first_pass_output['letter accuracy'], first_pass_output['word accuracy']]

if 'second pass model' in second_pass_output.keys():
	second_pass_output = second_pass_output['second pass model']
	second_pass_solution = deepcopy(second_pass_output['second pass model']['final grid']) # gives the grid
	second_pass_accu_list = [second_pass_output['final letter'], second_pass_output['final word']]


'''
	NOTE: take result output from above as you wish to...
'''

'''
	Drawing crossword grid for either first_pass_solution or second_pass_solution
'''
'''
  -------------------------------------------------------------------------------------------------------------------
'''
solution = first_pass_solution

# suitable conversion to the drawing grid format
for i in range(len(solution)):
	for j in range(len(solution[0])):
		if solution[i][j] == '':
			solution[i][j] = 0

overlay_truth_matrix = [[0] * cols for _ in range(rows)]
grid_num_matrix = [["-"] * cols for _ in range(rows)]
gold_grid_info = puzzle['grid']

wrong_clues_list = []

for i in range(rows):
	for j in range(cols):
		cell_info = gold_grid_info[i][j]
		cell_char = cell_info[1]
		cell_num = cell_info[0]
		if cell_num != 'BLACK':
			grid_num_matrix[i][j] = cell_num
		if solution[i][j] != cell_char and cell_info != 'BLACK':
			start_i = i
			start_j = j
			if cell_num == '':
				while start_j > -1:
					start_j -= 1
					if gold_grid_info[i][start_j][0] != '':
						wrong_clues_list.append(gold_grid_info[i][start_j][0] + " A")
						break
				while start_i > - 1:
					start_i -= 1
					if gold_grid_info[start_i][j][0] != '':
						wrong_clues_list.append(gold_grid_info[start_i][j][0]+ " D")
						break
			else:
				contained_grid_num = gold_grid_info[i][j][0]
				if contained_grid_num in puzzle['clues']['across'].keys():
					wrong_clues_list.append(contained_grid_num +" A")

				if contained_grid_num in puzzle['clues']['down'].keys():
					wrong_clues_list.append(contained_grid_num + " D")
			overlay_truth_matrix[i][j] = 1

wrong_A_num = [x.split(' ')[0] for x in list(set(wrong_clues_list)) if x.split(' ')[1] == 'A']
wrong_D_num = [x.split(' ')[0] for x in list(set(wrong_clues_list)) if x.split(' ')[1] == 'D']

wrong_clues = [wrong_A_num, wrong_D_num]
draw_grid(solution, [rows, cols], overlay_truth_matrix, grid_num_matrix, first_pass_accu_list, all_clue_info, wrong_clues, "06/09/2069", MODEL_TYPE)