class Variable():

    ACROSS = "across"
    DOWN = "down"

    def __init__(self, i, j, direction, length):
        """Create a new variable with starting point, direction, and length."""
        self.i = i
        self.j = j
        self.direction = direction
        self.length = length
        self.cells = []
        for k in range(self.length):
            self.cells.append(
                (self.i + (k if self.direction == Variable.DOWN else 0),
                 self.j + (k if self.direction == Variable.ACROSS else 0))
            )

    def __hash__(self):
        return hash((self.i, self.j, self.direction, self.length))

    def __eq__(self, other):
        return (
            (self.i == other.i) and
            (self.j == other.j) and
            (self.direction == other.direction) and
            (self.length == other.length)
        )

    def __str__(self):
        return f"({self.i}, {self.j}) {self.direction} : {self.length}"

    def __repr__(self):
        direction = repr(self.direction)
        return f"Variable({self.i}, {self.j}, {direction}, {self.length})"

class Crossword():
    def __init__(self, grid):
        self.structure = []

        self.height = len(grid) # the number of rows in the grid
        self.width = len(grid[0]) # the number of columns in the grid
        for i in range(len(grid)):
            row = []
            for j in range(len(grid[0])):
                if grid[i][j] == '':
                    row.append(False)
                else:
                    row.append(True)
            self.structure.append(row)
        # Determine variable set
        self.variables = set()

        for i in range(self.height):
            for j in range(self.width):

                # Vertical words
                starts_word = (
                    self.structure[i][j]
                    and (i == 0 or not self.structure[i - 1][j])
                )
                if starts_word:
                    length = 1
                    for k in range(i + 1, self.height):
                        if self.structure[k][j]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.DOWN,
                            length=length
                        ))

                # Horizontal words
                starts_word = (
                    self.structure[i][j]
                    and (j == 0 or not self.structure[i][j - 1])
                )
                if starts_word:
                    length = 1
                    for k in range(j + 1, self.width):
                        if self.structure[i][k]:
                            length += 1
                        else:
                            break
                    if length > 1:
                        self.variables.add(Variable(
                            i=i, j=j,
                            direction=Variable.ACROSS,
                            length=length
                        ))

def evaluate_grid(puzzle, solved_grid):
    grid_structure = []
    grid_solution = []

    for row_e in puzzle['grid']:
        row = []
        solution_row = []
        for col_e in row_e:
            if not isinstance(col_e, list):
                solution_row.append('')
                row.append('')
            else:
                solution_row.append(col_e[1])
                row.append('A')
        grid_structure.append(row)
        grid_solution.append(solution_row)
        
    crossword = Crossword(grid_structure)
    
    letters_correct = 0
    words_correct = 0

    total_letters = 0
    total_words = 0
    for slot in crossword.variables:
        total_words += 1
        ith_row, jth_col = slot.i, slot.j
        ans_len = slot.length
        ans_direction = slot.direction
        total_letters += ans_len
        temp_letter_count = 0

        if ans_direction == 'across':
            for k in range(jth_col, jth_col + ans_len):
                if grid_solution[ith_row][k] == solved_grid[ith_row][k]:
                    temp_letter_count += 1

        elif ans_direction == 'down':
            for k in range(ith_row, ith_row + ans_len):
                if grid_solution[k][jth_col] == solved_grid[k][jth_col]:
                    temp_letter_count += 1

        if temp_letter_count == ans_len:
            words_correct += 1

        letters_correct += temp_letter_count

    return letters_correct / total_letters, words_correct / total_words