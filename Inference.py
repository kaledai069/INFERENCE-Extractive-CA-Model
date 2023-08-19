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

parser = argparse.ArgumentParser(description="My Python Script")

parser.add_argument('--crossword_path', type=int, help='An integer argument')

args = parser.parse_args()

MODEL_PATH = "/content/Inference_components/dpr_biencoder_trained_500k.bin"
ANS_TSV_PATH = "/content/Inference_components/all_answer_list.tsv"
DENSE_EMBD_PATH = "/content/Inference_components/embeddings_all_answers_json_0*"

puzzle = json_CA_json_converter(args.crossword_path)
crossword = Crossword(puzzle)
solver = BPSolver(crossword, model_path = MODEL_PATH, ans_tsv_path = ANS_TSV_PATH, dense_embd_path = DENSE_EMBD_PATH, max_candidates = 10000)
solution = solver.solve(num_iters = 100, iterative_improvement_steps = 0)
print(solution)
solver.evaluate(solution)