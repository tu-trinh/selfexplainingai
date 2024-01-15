from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from constants import *
import numpy as np
import Enum
from enums import *
from typing import Union, List


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
    "drop": 4,
    "toggle": 5
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
    elif obj == "lava":
        desc = "a patch of lava"
    else:
        desc = "a " + color + " " + obj
    return desc

def get_unit_location_desc(desc, img_row, img_col, left = False, backwards = False, right = False):
    loc_desc = desc
    if left:
        x_diff = 4 - img_col
        y_diff = img_row - 2
    elif backwards:
        x_diff = img_row - 2
        y_diff = img_col - 4
    elif right:
        x_diff = img_col - 4
        y_diff = 2 - img_row
    else:
        x_diff = 2 - img_row
        y_diff = 4 - img_col
    if x_diff != 0 and y_diff != 0:
        if x_diff < 0:
            if abs(x_diff) == 1:
                loc_desc += f" {abs(x_diff)} cell right"
            else:
                loc_desc += f" {abs(x_diff)} cells right"
        else:
            if x_diff == 1:
                loc_desc += f" {x_diff} cell left"
            else:
                loc_desc += f" {x_diff} cells left"
        if y_diff < 0:
            if abs(y_diff) == 1:
                loc_desc += f" and {abs(y_diff)} cell behind"
            else:
                loc_desc += f" and {abs(y_diff)} cells behind"
        else:
            if y_diff == 1:
                loc_desc += f" and {y_diff} cell in front"
            else:
                loc_desc += f" and {y_diff} cells in front"
    elif x_diff == 0:
        if y_diff < 0:
            if abs(y_diff) == 1:
                loc_desc += f" {abs(y_diff)} cell behind"
            else:
                loc_desc += f" {abs(y_diff)} cells behind"
        else:
            if y_diff == 1:
                loc_desc += f" {y_diff} cell in front"
            else:
                loc_desc += f" {y_diff} cells in front"
    elif y_diff == 0:
        if x_diff < 0:
            if abs(x_diff) == 1:
                loc_desc += f" {abs(x_diff)} cell right"
            else:
                loc_desc += f" {abs(x_diff)} cells right"
        else:
            if x_diff == 1:
                loc_desc += f" {x_diff} cell left"
            else:
                loc_desc += f" {x_diff} cells left"
    return loc_desc

def get_obs_desc(obs, left_obs = None, backwards_obs = None, right_obs = None, detail = 3):
    """
    Detail levels:
    1 - list objects in the field of vision
    2 - list objects in the field of vision and their location
    3 - list what is in the field of vision row-by-row and directly at front/left/right
    4 - list everything cell by cell
    """
    if detail == 1:
        description = f"You are facing {IDX_TO_DIR[obs['direction']]}. Your field of vision is a {AGENT_VIEW_SIZE}x{AGENT_VIEW_SIZE} square in which you are located at the bottom middle. "
        img = obs["image"]
        objs = []
        obj_counts = {}
        for r in range(len(img)):
            for c in range(len(img[0])):
                if not (r == 2 and c == 4):  # agent's position
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        if desc not in obj_counts:
                            obj_counts[desc] = 1
                        else:
                            obj_counts[desc] += 1
                        objs.append(desc + " " + str(obj_counts[desc]))
        if len(objs) > 0:
            if len(objs) == 1:
                description += f"You see {objs[0]}."
            elif len(objs) == 2:
                description += f"You see {objs[0]} and {objs[1]}."
            else:
                description += f"You see {', '.join(objs[:-1])}, and {objs[-1]}."
        direct_front = get_unit_desc(img[2][3])
        if "unknown" not in direct_front and "floor" not in direct_front:
            description += f" In the cell directly in front of you is the {' '.join(direct_front.split()[1:])}."
        carrying_obj = get_unit_desc(img[2][4])
        if "unknown" not in carrying_obj and "floor" not in carrying_obj:
            description += f" Finally, you are carrying {carrying_obj}."
        return description
    if detail == 2:
        description = f"From your exploration, you know that you are facing {IDX_TO_DIR[obs['direction']]}. Around you, you see the following:"
        img = obs["image"]
        objs = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                if not (r == 2 and c == 4):  # agent's position
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c))
        if left_obs is not None:
            img = left_obs["image"]
            for r in range(len(img)):
                for c in range(2):  # leave out the center overlaps
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c, left = True))
        if backwards_obs is not None:
            img = backwards_obs["image"]
            for r in range(len(img)):
                for c in range(len(img[0]) - 1):  # leave out the overlapping row
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c, backwards = True))
        if right_obs is not None:
            img = right_obs["image"]
            for r in range(len(img)):
                for c in range(2):  # leave out the center overlaps
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c, right = True))
        if len(objs) > 0:
            if len(objs) == 1:
                description += f" {objs[0]}."
            elif len(objs) == 2:
                description += f" {objs[0]} and {objs[1]}."
            else:
                description += f" {', '.join(objs[:-1])}, and {objs[-1]}."
        carrying_obj = get_unit_desc(img[2][4])
        if "unknown" not in carrying_obj and "floor" not in carrying_obj:
            description += f" Finally, you are carrying {carrying_obj}."
        return description
    if detail == 3:
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

def list_available_actions(scenario):
    if scenario == 1:
        actions = """6. Go to key (only if you have seen one).\n7. Go to door (only if you have seen one).\n8. Go to goal (only if you have seen one).\n9. Pick up key (only if you have seen one).\n10. Open door (only if you have seen one)."""
    elif scenario == 2:
        actions = """6. Go to goal (only if you have seen one)."""
    elif scenario == 3:
        actions = """6. Go to goal (only if you have seen one)."""
    elif scenario == 4:
        actions = """6. Go to door (only if you have seen one).\n7. Go to goal (only if you have seen one).\n8. Open door (only if you have seen one)."""
    return actions

def convert_response_to_action(resp, additional_actions):
    return resp

    # Only needed for automation
    # if "forward" in resp or "1" in resp:
    #     return 1
    # if "left" in resp or "2" in resp:
    #     return 2
    # if "right" in resp or "3" in resp:
    #     return 3
    # if "pick" in resp or "4" in resp:
    #     return 4
    # if "put" in resp or "5" in resp:
    #     return 5
    # if "unlock" in resp or "6" in resp:
    #     return 6
    # if "open" in resp or "7" in resp:
    #     return 7
    # if "close" in resp or "8" in resp:
    #     return 8
    # return None

def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x2 - x1) + abs(y2 - y1)

def get_adjacent_cells(cell):
    x, y = cell
    return set([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

def convert_to_enum(enum: Enum, value: Union[List, str]):
    if isinstance(value, str):
        return enum[value]
    return [enum[val] for val in value]
    