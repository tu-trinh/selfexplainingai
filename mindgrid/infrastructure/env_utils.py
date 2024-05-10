from collections import deque
from typing import Dict, List, Union

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

from mindgrid.infrastructure.env_constants import (AGENT_VIEW_SIZE, DIR_TO_VEC,
                                                   IDX_TO_COLOR, IDX_TO_DIR,
                                                   IDX_TO_OBJECT, IDX_TO_STATE,
                                                   NUM_TO_ORDERING,
                                                   NUM_TO_WORD)


def get_unit_desc(cell: Union[np.ndarray, List]) -> str:
    """
    Gets descriptor phrase for a singular cell
    """
    obj = IDX_TO_OBJECT[cell[0]]
    color = IDX_TO_COLOR[cell[1]]
    if obj == "door":
        state = IDX_TO_STATE[cell[2]]
        beginning = "an" if state == "open" else "a"
        desc = beginning + " " + state + " " + color + " " + obj
    elif obj == "empty" or obj == "floor":
        desc = "empty floor"
    elif obj == "unseen":
        desc = "an unknown cell"
    elif obj == "lava":
        desc = "a patch of lava"
    elif obj == "agent":
        desc = "agent"
    else:
        desc = "a " + color + " " + obj
    return desc


def get_unit_location_desc(
    desc: str,
    img_row: int,
    img_col: int,
    left: bool = False,
    backwards: bool = False,
    right: bool = False,
) -> str:
    """
    Gets location-grounded descriptor phrase for a singular cell
    """
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


def get_obs_desc(
    obs: Dict,
    left_obs: Dict = None,
    backwards_obs: Dict = None,
    right_obs: Dict = None,
    detail: int = 3,
    carrying: WorldObj = None,
) -> str:
    """
    Detail levels:
    0 - TODO: list objects in the field of vision, grouped
    1 - list objects in the field of vision, individually
    2 - list objects in the 360 field of vision and their location
    3 - list what is in the field of vision row-by-row and directly at front/left/right
    4 - list everything cell by cell, excluding border walls
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
                        objs.append(get_unit_location_desc(desc, r, c, left=True))
        if backwards_obs is not None:
            img = backwards_obs["image"]
            for r in range(len(img)):
                for c in range(len(img[0]) - 1):  # leave out the overlapping row
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c, backwards=True))
        if right_obs is not None:
            img = right_obs["image"]
            for r in range(len(img)):
                for c in range(2):  # leave out the center overlaps
                    desc = get_unit_desc(img[r][c])
                    if "unknown" not in desc and "floor" not in desc:
                        objs.append(get_unit_location_desc(desc, r, c, right=True))
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

        for viewed_row in range(
            AGENT_VIEW_SIZE - 2, -1, -1
        ):  # this is actually a column in the img
            objs = np.array(
                [get_unit_desc(img[i][viewed_row]) for i in range(AGENT_VIEW_SIZE)]
            )
            if all(objs == "an unknown cell"):
                if viewed_row == AGENT_VIEW_SIZE - 2:
                    description += (
                        f"You cannot see what is in the row in front of you. "
                    )
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
                                row_view[-1] = " ".join(row_view[-1].split()[1:]) + "s"
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
    if detail == 4:
        img = obs["image"]
        size = img.shape[-2]
        description = f"Description of room (sized {size - 2} x {size - 2}), going from leftmost column (Col 1) to rightmost column (Col {size - 2}), top cell to bottom cell for each column:\n"

        for c in range(1, len(img) - 1):  # skip leftmost and rightmost columns of walls
            description += f"Col {c}: "
            for r in range(1, len(img[0]) - 1):  # skip top and bottom walls in column
                desc = get_unit_desc(img[r][c])
                if desc == "empty floor":
                    actual_desc = "floor"
                elif desc == "agent":
                    if carrying is not None:
                        carry_addendum = (
                            f" and carrying a {carrying.color} {carrying.type}"
                        )
                    else:
                        carry_addendum = ""
                    actual_desc = f"your location (you're facing {IDX_TO_DIR[obs['direction']]}{carry_addendum})"
                else:
                    actual_desc = desc
                if r == len(img[0]) - 2:
                    description += "and " + actual_desc + "\n"
                else:
                    description += actual_desc + ", "
        return description.strip()


def get_babyai_desc(env: MiniGridEnv, image: np.ndarray) -> str:
    """
    Get BabyAI-Text style environment description
    """
    list_textual_descriptions = []
    if env.carrying is not None:
        list_textual_descriptions.append(
            "You carry a {} {}".format(env.carrying.color, env.carrying.type)
        )

    agent_pos_vx, agent_pos_vy = env.get_view_coords(env.agent_pos[0], env.agent_pos[1])

    view_field_dictionary = dict()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                if i not in view_field_dictionary.keys():
                    view_field_dictionary[i] = dict()
                    view_field_dictionary[i][j] = image[i][j]
                else:
                    view_field_dictionary[i][j] = image[i][j]

    # Find the wall if any
    #  We describe a wall only if there is no objects between the agent and the wall in straight line

    # Find wall in front
    j = agent_pos_vy - 1
    object_seen = False
    while j >= 0 and not object_seen:
        if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
            if image[agent_pos_vx][j][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vy - j} step{'s' if agent_pos_vy - j > 1 else ''} forward"
                )
                object_seen = True
            else:
                object_seen = True
        j -= 1
    # Find wall left
    i = agent_pos_vx - 1
    object_seen = False
    while i >= 0 and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {agent_pos_vx - i} step{'s' if agent_pos_vx - i > 1 else ''} left"
                )
                object_seen = True
            else:
                object_seen = True
        i -= 1
    # Find wall right
    i = agent_pos_vx + 1
    object_seen = False
    while i < image.shape[0] and not object_seen:
        if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
            if image[i][agent_pos_vy][0] == 2:
                list_textual_descriptions.append(
                    f"You see a wall {i - agent_pos_vx} step{'s' if i - agent_pos_vx > 1 else ''} right"
                )
                object_seen = True
            else:
                object_seen = True
        i += 1

    # returns the position of seen objects relative to you
    for i in view_field_dictionary.keys():
        for j in view_field_dictionary[i].keys():
            if i != agent_pos_vx or j != agent_pos_vy:
                object = view_field_dictionary[i][j]
                relative_position = dict()

                if i - agent_pos_vx > 0:
                    relative_position["x_axis"] = ("right", i - agent_pos_vx)
                elif i - agent_pos_vx == 0:
                    relative_position["x_axis"] = ("face", 0)
                else:
                    relative_position["x_axis"] = ("left", agent_pos_vx - i)
                if agent_pos_vy - j > 0:
                    relative_position["y_axis"] = ("forward", agent_pos_vy - j)
                elif agent_pos_vy - j == 0:
                    relative_position["y_axis"] = ("forward", 0)

                distances = []
                if relative_position["x_axis"][0] in ["face", "en face"]:
                    distances.append(
                        (relative_position["y_axis"][1], relative_position["y_axis"][0])
                    )
                elif relative_position["y_axis"][1] == 0:
                    distances.append(
                        (relative_position["x_axis"][1], relative_position["x_axis"][0])
                    )
                else:
                    distances.append(
                        (relative_position["x_axis"][1], relative_position["x_axis"][0])
                    )
                    distances.append(
                        (relative_position["y_axis"][1], relative_position["y_axis"][0])
                    )

                description = ""
                if object[0] != 4:  # if it is not a door
                    description = f"You see a {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "

                else:
                    if IDX_TO_STATE[object[2]] != 0:  # if it is not open
                        description = f"You see a {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "

                    else:
                        description = f"You see an {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "

                for _i, _distance in enumerate(distances):
                    if _i > 0:
                        description += " and "

                    description += f"{_distance[0]} step{'s' if _distance[0] > 1 else ''} {_distance[1]}"

                list_textual_descriptions.append(description)
    return ". ".join(list_textual_descriptions) + "."


def get_full_env_desc(full_obs: np.ndarray) -> str:
    """
    Describe full environment
    """
    actual_room_size = full_obs.shape[0] - 2
    description = f"The environment is a {actual_room_size}x{actual_room_size} square. From top to bottom, it has the following:\n"
    for row in range(1, actual_room_size + 1):
        description += f"{NUM_TO_ORDERING[row]} row: "
        row_objs = []
        for col in range(1, actual_room_size + 1):
            cell = full_obs[row][col]
            obj_desc = get_unit_desc(cell)
            if "floor" not in obj_desc and "unknown" not in obj_desc:
                row_objs.append(obj_desc)
        description += ", ".join(row_objs) + "\n"
    return description.strip()


def bfs(grid, start_dir, start_pos, end_pos):

    """
    for j in range(grid.shape[1]):
        for i in range(grid.shape[0]):
            print(grid[i][j], end=' ')
        print()
    """

    state = (start_pos, start_dir)
    queue = deque([state])
    trace_back = {}
    trace_back[state] = -1

    while queue:
        state = queue.popleft()
        (x, y), dir = state

        if (x, y) in end_pos:
            actions = []
            while trace_back[state] != -1:
                state, action = trace_back[state]
                actions.append(action)
            return list(reversed(actions))

        # forward
        dir_vec = DIR_TO_VEC[dir]
        nx, ny = x + dir_vec[0], y + dir_vec[1]
        nstate = ((nx, ny), dir)
        if grid[nx, ny] == 0 and nstate not in trace_back:
            queue.append(nstate)
            trace_back[nstate] = (state, Actions.forward)

        # rotate
        for d in [-1, 1]:
            ndir = (dir + d + 4) % 4
            nstate = ((x, y), ndir)
            if nstate not in trace_back:
                queue.append(nstate)
                trace_back[nstate] = (state, Actions.left if d == -1 else Actions.right)

    return None
