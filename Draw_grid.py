import numpy as np
import cv2


def draw_grid(data, overlay_truth_matrix, grid_num_matrix, accu_list, all_clue_info, wrong_clues):
    rows, cols = 15, 15
    cell_size = 38
    padding_w = 340

    BOX_OFFSET = 35

    wrong_A_num, wrong_D_num = wrong_clues

    if len(all_clue_info[0]) > 40 or len(all_clue_info[1]) > 40:
        padding_h = 120
    else:
        padding_h = 80

    width = cols * cell_size + 2 * padding_w
    height = rows * cell_size + 2 * padding_h

    image = np.ones((height, width, 3), dtype=np.uint8) * 255


    for i in range(rows):
        for j in range(cols):
            cell_value = data[i][j]
            cell_x = j * cell_size + padding_w
            cell_y = i * cell_size + padding_h - BOX_OFFSET

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            font_thickness = 1
            
            if cell_value == 0:
                cv2.rectangle(image, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), -1)  # Fill the cell with black
            elif overlay_truth_matrix[i][j] == 1:
                text_size = cv2.getTextSize(cell_value, font, font_scale, font_thickness)[0]
                text_x = cell_x + (cell_size - text_size[0]) // 2
                text_y = cell_y + (cell_size + text_size[1]) // 2
                cv2.rectangle(image, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (63, 27, 196), -1)
                cv2.putText(image, cell_value, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            else:
                text_size = cv2.getTextSize(cell_value, font, font_scale, font_thickness)[0]
                text_x = cell_x + (cell_size - text_size[0]) // 2
                text_y = cell_y + (cell_size + text_size[1]) // 2
                cv2.putText(image, cell_value, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            if grid_num_matrix[i][j] != '-':
                grid_num = grid_num_matrix[i][j]
                
                grid_num_x = cell_x + 2
                grid_num_y = cell_y + 10
                cv2.putText(image, grid_num, (grid_num_x, grid_num_y), font, 0.28, (0, 0, 0, 96), 1, cv2.LINE_AA)

    letter_accuracy_text = f"Letter Accuracy: {accu_list[0]:.2f} %"
    text_size = cv2.getTextSize(letter_accuracy_text, font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = 340
    t_y = 30
    cv2.putText(image, letter_accuracy_text, (t_x, t_y), font, 0.65, (0, 0, 0), font_thickness, cv2.LINE_AA)

    word_accuracy_text = f"Word Accuracy: {accu_list[1]:.2f} %"
    text_size = cv2.getTextSize(word_accuracy_text, font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = width // 2 + 25
    t_y = 30
    cv2.putText(image, word_accuracy_text, (t_x, t_y), font, 0.65, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # text for 'across'
    text_size = cv2.getTextSize("ACROSS", font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = 25
    t_y = 40
    cv2.putText(image, "ACROSS", (t_x, t_y), font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    # down text
    text_size = cv2.getTextSize("DOWN", font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = 950
    t_y = 40
    cv2.putText(image, "DOWN", (t_x, t_y), font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    text_limit = 500

    y_start_ind = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(all_clue_info[0])):
        clue_text = all_clue_info[0][i][0] + '. ' + all_clue_info[0][i][1]

        if all_clue_info[0][i][0] in wrong_A_num:
            color = (0, 0, 128)
        else:
            color = (0, 0, 0)

        start_index_size = cv2.getTextSize(all_clue_info[0][i][0] + '. ', font, font_scale, font_thickness)[0][0]
        text_size = cv2.getTextSize(clue_text, font, font_scale, font_thickness)[0]
        if text_size[0] > text_limit:
            multiples = (text_size[0] // text_limit) + 1

            for i in range(multiples):
                if i != 0:
                    t_x = 15 + start_index_size - 20
                else:
                    t_x = 15
                t_y = 60 + y_start_ind * 15
                y_start_ind += 1
                ex_clue_text = clue_text[i * 45: (i+1) * 45]
                cv2.putText(image, ex_clue_text, (t_x, t_y), font, 0.40, color, 1, cv2.LINE_AA)
        else: 
            t_x = 15
            t_y = 60 + y_start_ind * 15
            cv2.putText(image, clue_text, (t_x, t_y), font, 0.40, color, 1, cv2.LINE_AA)
            y_start_ind += 1

    y_start_ind = 0
    for i in range(len(all_clue_info[1])):
        clue_text = all_clue_info[1][i][0] + '. ' + all_clue_info[1][i][1]
        if all_clue_info[1][i][0] in wrong_D_num:
            color = (0, 0, 128)
        else:
            color = (0, 0, 0)
        start_index_size = cv2.getTextSize(all_clue_info[1][i][0] + '. ', font, font_scale, font_thickness)[0][0]
        text_size = cv2.getTextSize(clue_text, font, font_scale, font_thickness)[0]
        if text_size[0] > text_limit:
            multiples = (text_size[0] // text_limit) + 1

            for i in range(multiples):
                if i != 0:
                    t_x = 930 + start_index_size - 20
                else:
                    t_x = 930
                t_y = 60 + y_start_ind * 15
                y_start_ind += 1
                ex_clue_text = clue_text[i * 45: (i+1) * 45]
                cv2.putText(image, ex_clue_text, (t_x, t_y), font, 0.40, color, 1, cv2.LINE_AA)
        else: 
            t_x = 930
            t_y = 60 + y_start_ind * 15
            cv2.putText(image, clue_text, (t_x, t_y), font, 0.40, color, 1, cv2.LINE_AA)
            y_start_ind += 1


    for i in range(rows + 1):
        y = i * cell_size + padding_h - BOX_OFFSET
        cv2.line(image, (padding_w, y), (width - padding_w, y), (0, 0, 0), 1)

    for j in range(cols + 1):
        x = j * cell_size + padding_w 
        cv2.line(image, (x, padding_h - BOX_OFFSET), (x, height - padding_h - BOX_OFFSET), (0, 0, 0), 1)

    # Draw a border around the grid
    border_thickness = 2  # You can adjust this as needed
    cv2.rectangle(image, (padding_w, padding_h - BOX_OFFSET), (width - padding_w - 1, height - padding_h - 1 - BOX_OFFSET), (0, 0, 0), border_thickness)

    # Display the grid with characters, padding, and inner grid lines
    cv2.imshow('Solved Crossword', image)
    cv2.imwrite('./solved_crosswords/crossword_TODAY.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()