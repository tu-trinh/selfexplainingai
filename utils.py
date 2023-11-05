from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from constants import *
import numpy as np


IDX_TO_STATE = {
    0: "open",
    1: "closed",
    2: "locked"
}
IDX_TO_DIR = {
    0: "east",
    1: "south",
    2: "west",
    3: "north"
}
ACTION_TO_IDX = {
    "left": 0,
    "right": 1, 
    "forward": 2,
    "pickup": 3, 
    "putdown": 4,
    "unlock": 5
}
CUSTOM_ACTION_TO_TRUE_ACTION = {
    1: 2,
    2: 0,
    3: 1,
    4: 3,
    5: 4,
    6: 5,
    7: 5,
    8: 5
}
NUM_TO_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five"
}

def get_unit_desc(cell):
    obj = IDX_TO_OBJECT[cell[0]]
    color = IDX_TO_COLOR[cell[1]]
    if obj == "door":
        state = IDX_TO_STATE[cell[2]]
        beginning = "an" if state == "open" else "a"
        desc = beginning + " " + state + " " + color + " " + obj
    elif obj == "empty":
        desc = "empty floor"
    elif obj == "unseen":
        desc = "an unknown cell"
    else:
        desc = "a " + color + " " + obj
    return desc

def get_obs_desc(obs):
    description = f"You are facing {IDX_TO_DIR[obs['direction']]}. Your field of vision is a {AGENT_VIEW_SIZE}x{AGENT_VIEW_SIZE} square in which you are located at the bottom middle. In the following description, an \"unknown cell\" is one for which your vision is blocked, so you can't tell what is there. "
    img = obs["image"]
    
    direct_left = get_unit_desc(img[1][4])
    farther_left = get_unit_desc(img[0][4])
    direct_right = get_unit_desc(img[3][4])
    farther_right = get_unit_desc(img[4][4])
    if farther_left == "an unknown cell":
        description += f"Directly to your left is {direct_left}, blocking your vision such that you can't see the cell to its left. "
    else:
        # if direct_left != "empty floor":
        description += f"Directly to your left is {direct_left}. "
        # if farther_left != "empty floor":
        description += f"Two cells to your left is {farther_left}. "
    if farther_right == "an unknown cell":
        description += f"Directly to your right is {direct_right}, blocking your vision such that you can't see the cell to its right. "
    else:
        # if direct_right != "empty floor":
        description += f"Directly to your right is {direct_right}. "
        # if farther_right != "empty floor":
        description += f"Two cells to your right is {farther_right}. "
    
    direct_front = get_unit_desc(img[2][3])
    
    for viewed_row in range(AGENT_VIEW_SIZE - 2, -1, -1):  # this is actually a column in the img
        objs = np.array([get_unit_desc(img[i][viewed_row]) for i in range(AGENT_VIEW_SIZE)])
        if all(objs == "an unknown cell"):
            if viewed_row == AGENT_VIEW_SIZE - 2:
                description += f"You cannot see what is in the row in front of you. "
            else:
                description += f"You cannot see {AGENT_VIEW_SIZE - 1 - viewed_row} rows in front of you. "
        # elif not all(objs == "empty floor"):
        else:
            row_view = []
            processed_set = set()
            prev_obj = objs[0]
            counter = 1
            row_view.append(prev_obj)
            for i in range(1, len(objs)):
                if objs[i] == prev_obj:
                    counter += 1
                    if objs[i] not in processed_set:
                        if "floor" not in row_view[-1]:
                            row_view[-1]  = " ".join(row_view[-1].split()[1:]) + "s"
                        processed_set.add(objs[i])
                elif objs[i] != prev_obj:
                    if counter > 1:
                        row_view[-1] += f" for {NUM_TO_WORD[counter]} cells"
                    row_view.append(objs[i])  # w w w n w
                    counter = 1
                prev_obj = objs[i]
            if counter > 1:
                row_view[-1] += f" for {NUM_TO_WORD[counter]} cells"
            num_rows_in_front = NUM_TO_WORD[AGENT_VIEW_SIZE - 1 - viewed_row]
            row_word = "row" if num_rows_in_front == "one" else "rows"
            if len(row_view) > 1:
                description += f"{num_rows_in_front.capitalize()} {row_word} in front of you, from left to right, you see {', '.join(row_view[:-1])}, and {row_view[-1]}"
            else:
                description += f"{num_rows_in_front.capitalize()} {row_word} in front of you, from left to right, you see {row_view[-1]}"
            if num_rows_in_front == "one":
                description += f" (the {' '.join(direct_front.split()[1:])} is directly in front of you). "
            else:
                description += ". "
    
    agent_obj = get_unit_desc(img[2][4])
    if agent_obj != "an unknown cell" and "floor" not in agent_obj:
        description += f"Finally, you are holding {agent_obj}."
    
    return description
