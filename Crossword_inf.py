from Utils_inf import clean

class Crossword:
    def __init__(self, data):
        self.initialize_grids(grid=data["grid"])
        self.initialize_clues(clues=data["clues"])
        self.initialize_variables()

    def initialize_grids(self, grid):
        self.letter_grid = [[grid[j][i][1] if type(grid[j][i]) == list else "" for i in
                             range(len(grid[0]))] for j in range(len(grid))]
        self.number_grid = [[grid[j][i][0] if type(grid[j][i]) == list else "" for i in
                             range(len(grid[0]))] for j in range(len(grid))]
        self.grid_cells = {}

    def initialize_clues(self, clues):
        self.across = clues["across"]
        self.down = clues["down"]

    def initialize_variable(self, position, clues, across=True):
        row, col = position
        cell_number = self.number_grid[row][col]
        assert cell_number in clues, print("Missing clue")
        word_id = cell_number + "A" if across else cell_number + "D"
        clue = clean(clues[cell_number][0])
        answer = clean(clues[cell_number][1])
        for idx in range(len(answer)):
            cell = (row, col + idx) if across else (row + idx, col)
            if cell in self.grid_cells:
                self.grid_cells[cell].append(word_id)
            else:
                self.grid_cells[cell] = [word_id]
            if word_id in self.variables:
                self.variables[word_id]["cells"].append(cell)
            else:
                self.variables[word_id] = {"clue": clue, "gold": answer, "cells": [cell], "crossing": []}

    def initialize_crossing(self):
        for word_id in self.variables:
            cells = self.variables[word_id]["cells"]
            crossing_ids = []
            for cell in cells:
                crossing_ids += list(filter(lambda x: x!= word_id, self.grid_cells[cell]))
            self.variables[word_id]["crossing"] = crossing_ids

    def initialize_variables(self):
        self.variables = {}
        for row in range(len(self.number_grid)):
            for col in range(len(self.number_grid[0])):
                cell_number = self.number_grid[row][col]
                if self.number_grid[row][col] != "":
                    if cell_number in self.across:
                        self.initialize_variable((row, col), self.across, across=True)
                    if cell_number in self.down:
                        self.initialize_variable((row, col), self.down, across=False)
        self.initialize_crossing()