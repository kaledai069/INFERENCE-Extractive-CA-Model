import pandas as pd
import re
import tokenizers
import json
import puz
import os
import numpy as np
import streamlit as st
import scipy
import sys
import subprocess
import copy
import json
from itertools import zip_longest
from copy import deepcopy
import regex
from Crossword_inf import Crossword
from BPSolver_inf import BPSolver
from Models_inf import setup_closedbook, DPRForCrossword
from Utils_inf import print_grid
from Normal_utils_inf import puz_to_json
from Strict_json import json_CA_json_converter
import argparse
import time
from Draw_grid import draw_grid
import requests
import json
import datetime


MODEL_CONFIG = {
	'bert': {
		'MODEL_PATH' : "./Inference_components/dpr_biencoder_trained_EPOCH_2_COMPLETE.bin",
		'ANS_TSV_PATH': "./Inference_components/all_answer_list.tsv",
		'DENSE_EMBD_PATH': "./Inference_components/embeddings_BERT_EPOCH_2_COMPLETE0.pkl"
	},
	'distilbert': {
		'MODEL_PATH': "./Inference_components/distilbert_EPOCHs_7_COMPLETE.bin", 
		'ANS_TSV_PATH': "./Inference_components/all_answer_list.tsv",
		'DENSE_EMBD_PATH': "./Inference_components/distilbert_7_epochs_embeddings.pkl"
	}
}


def getGrid(dateStr):
    headers = {
        'Referer': 'https://www.xwordinfo.com/JSON/'
    }
    # mm/dd/yyyy
    url = 'https://www.xwordinfo.com/JSON/Data.ashx?date=' + dateStr

    response = requests.get(url, headers=headers)

    context = {}
    grid_data = {}
    if response.status_code == 200:
        bytevalue = response.content
        jsonText = bytevalue.decode('utf-8').replace("'", '"')
        grid_data = json.loads(jsonText)
        return grid_data
    else:
        print(f"Request failed with status code {response.status_code}.")

parser = argparse.ArgumentParser(description="My Python Script")
parser.add_argument('--crossword_path', default = "nothing", type=str, help='Path to crossword JSON file.')
parser.add_argument('--date', type = str, help = 'Crossdate to inference to.')
parser.add_argument('--model', type = str, default = "bert", help = "Model type to inference with.")
args = parser.parse_args()

MODEL_TYPE = vars(args)['model']
DATE = vars(args)['date']

# Run the model off the 'DATE' -> "20XX/XX/XX" or from the .puz Crossword data file. 
if args.date:
	puzzle = getGrid(args.date)
	puzzle = json_CA_json_converter(puzzle, False)
else:
	puzzle = json_CA_json_converter(args.crossword_path, True)

for dim in ['across', 'down']:
	for grid_num in puzzle['clues'][dim].keys():
		clue_answer_list = puzzle['clues'][dim][grid_num]
		clue_section = clue_answer_list[0]
		ans_section = clue_answer_list[1]
		clue_section = clue_section.replace("&quot;", "").replace("&#39;", "").replace("<em>", "").replace("</em>", "")
		puzzle['clues'][dim][grid_num] = [clue_section, ans_section]

all_clues = puzzle['clues']

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

crossword = Crossword(puzzle)
start_time = time.time()

choosen_model_path = MODEL_CONFIG[MODEL_TYPE]['MODEL_PATH']
ans_list_path = MODEL_CONFIG[MODEL_TYPE]['ANS_TSV_PATH']
dense_embedding_path = MODEL_CONFIG[MODEL_TYPE]['DENSE_EMBD_PATH']


# print(choosen_model_path, ans_list_path, dense_embedding_path)
try: 
	solver = BPSolver(crossword, model_path = choosen_model_path, ans_tsv_path = ans_list_path, dense_embd_path = dense_embedding_path, max_candidates = 20000, model_type = MODEL_TYPE)
	solution = solver.solve(num_iters = 60, iterative_improvement_steps = 0)
	accu_list = solver.evaluate(solution)
except: 
	print("Error Occured for date: ", args.date)
	accu_list = []

end_time = time.time()
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


# suitable conversion to the drawing grid format
for i in range(len(solution)):
	for j in range(len(solution[0])):
		if solution[i][j] == '':
			solution[i][j] = 0

rows = 15
cols = 15

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
# accu_list = [69, 59]

wrong_clues = [wrong_A_num, wrong_D_num]
draw_grid(solution, overlay_truth_matrix, grid_num_matrix, accu_list, all_clue_info, wrong_clues, DATE, MODEL_TYPE)