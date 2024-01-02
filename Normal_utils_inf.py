import puz
import json
import requests

def puz_to_json(fname):
    """ Converts a puzzle in .puz format to .json format
    """
    p = puz.read(fname)
    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index:cell + col_index + 1] == '.':
                grid[row_idx][col_index] = 'BLACK'
            else:
                grid[row_idx][col_index] = ["", row_solution[col_index: col_index + 1]]

    across_clues = {}
    for clue in numbering.across:
        answer = ''.join(p.solution[clue['cell'] + i] for i in range(clue['len']))
        across_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])

    down_clues = {}
    for clue in numbering.down:
        answer = ''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len']))
        down_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])


    mydict = {'metadata': {'date': None, 'rows': p.height, 'cols': p.width}, 'clues': {'across': across_clues, 'down': down_clues}, 'grid': grid}
    return mydict

def puz_to_pairs(filepath):
    """ Takes in a filepath pointing to a .puz file and returns a list of (clue, fill) pairs in a list
    """
    p = puz.read(filepath)

    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index:cell + col_index + 1] == '.':
                grid[row_idx][col_index] = 'BLACK'
            else:
                grid[row_idx][col_index] = ["", row_solution[col_index: col_index + 1]]

    pairs = {}
    for clue in numbering.across:
        answer = ''.join(p.solution[clue['cell'] + i] for i in range(clue['len']))
        pairs[clue['clue']] = answer

    for clue in numbering.down:
        answer = ''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len']))
        pairs[clue['clue']] = answer

    return [(k, v) for k, v in pairs.items()]

def json_CA_json_converter(json_file_path, is_path):
    try:
        if is_path:
            with open(json_file_path, "r") as file:
                data = json.load(file)
        else:
            data = json_file_path

        json_conversion_dict = {}

        rows = data["size"]["rows"]
        cols = data["size"]["cols"]
        date = data["date"]

        clues = data["clues"]
        answers = data["answers"]

        json_conversion_dict["metadata"] = {"date": date, "rows": rows, "cols": cols}

        across_clue_answer = {}
        down_clue_answer = {}

        for clue, ans in zip(clues["across"], answers["across"]):
            split_clue = clue.split(" ")
            clue_num = split_clue[0][:-1]
            clue_ = " ".join(split_clue[1:])
            clue_ = clue_.replace("[", "").replace("]", "")
            across_clue_answer[clue_num] = [clue_, ans]

        for clue, ans in zip(clues["down"], answers["down"]):
            split_clue = clue.split(" ")
            clue_num = split_clue[0][:-1]
            clue_ = " ".join(split_clue[1:])
            clue_ = clue_.replace("[", "").replace("]", "")
            down_clue_answer[clue_num] = [clue_, ans]

        json_conversion_dict["clues"] = {
            "across": across_clue_answer,
            "down": down_clue_answer,
        }

        grid_info = data["grid"]
        grid_num = data["gridnums"]

        grid_info_list = []
        for i in range(rows):
            row_list = []
            for j in range(cols):
                if grid_info[i * rows + j] == ".":
                    row_list.append("BLACK")
                else:
                    if grid_num[i * rows + j] == 0:
                        row_list.append(["", grid_info[i * rows + j]])
                    else:
                        row_list.append(
                            [str(grid_num[i * rows + j]), grid_info[i * rows + j]]
                        )
            grid_info_list.append(row_list)

        json_conversion_dict["grid"] = grid_info_list

        return json_conversion_dict
    
    except:
        print("ERROR has occured.")

def fetch_nyt_crossword(dateStr):
    '''
        Fetch NYT puzzle from a specific date.
    '''

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
        puzzle_data = json_CA_json_converter(grid_data, False)
        for dim in ['across', 'down']:
            for grid_num in puzzle_data['clues'][dim].keys():
                clue_answer_list = puzzle_data['clues'][dim][grid_num]
                clue_section = clue_answer_list[0]
                ans_section = clue_answer_list[1]
                clue_section = clue_section.replace("&quot;", "'").replace("&#39;", "'")
                puzzle_data['clues'][dim][grid_num] = [clue_section, ans_section]
        return puzzle_data

    else:
        print(f"Request failed with status code {response.status_code}.")

