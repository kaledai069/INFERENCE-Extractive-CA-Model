import re
from collections import defaultdict
import string
import requests
import json
from scipy.special import softmax
import numpy as np
from Models_inf import answer_clues, setup_closedbook
from Strict_json import json_CA_json_converter
from Crossword_inf import Crossword


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

DATE = "10/01/2023"
puzzle = getGrid(DATE)
puzzle = json_CA_json_converter(puzzle, False)


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

all_clues = []
for var in crossword.variables:
    all_clues.append(crossword.variables[var]['clue'])
	

MODEL_PATH = "./Inference_components/distilbert_EPOCHs_7_COMPLETE.bin", 
ANS_TSV_PATH = "./Inference_components/all_answer_list.tsv",
DENSE_EMBD_PATH = "./Inference_components/distilbert_7_epochs_embeddings.pkl"
MODEL_TYPE = "distilbert"
dpr = setup_closedbook(MODEL_PATH, ANS_TSV_PATH, DENSE_EMBD_PATH, 0, MODEL_TYPE)
all_words, all_scores = answer_clues(dpr, all_clues, max_answers = 20000, output_strings=True) 
print(len(all_words))
print(all_words[0])
# print(dir(all_words))
# print(dir(all_scores))