import re
import random
import inflect
from collections import deque
from typing import Dict, List, Union
from copy import deepcopy as dc
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

from mindgrid.infrastructure.env_constants import (
    AGENT_VIEW_SIZE,
    DIR_TO_VEC,
    IDX_TO_COLOR,
    IDX_TO_DIR,
    IDX_TO_OBJECT,
    IDX_TO_STATE,
    NUM_TO_ORDERING,
    NUM_TO_WORD,
    OBJECT_TO_IDX,
)


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


def are_objects_equal(o, oo):
    if o is None and oo is None:
        return True
    if o is None or oo is None:
        # print("none")
        return False
    if type(o) != type(oo):
        # print("type")
        return False
    if o.color != oo.color:
        # print("color")
        return False
    if not are_objects_equal(o.contains, oo.contains):
        # print("contains")
        return False
    if o.init_pos != oo.init_pos:
        # print("init_pos")
        return False
    if o.cur_pos != oo.cur_pos:
        # print("cur_pos", o.cur_pos, oo.cur_pos)
        return False
    return True


def bfs(grid, start_dir, start_pos, end_pos):

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


def relative_position(dir, point):

    dx, dy = DIR_TO_VEC[dir]
    x, y = point

    # Determine the direction based on the vector
    if dx == 0 and dy == -1:
        cardinal = "north"
        front_back = "in front" if y < 0 else "behind"
        left_right = "to the right" if x > 0 else "to the left"
        return left_right, front_back
    elif dx == 0 and dy == 1:
        cardinal = "south"
        front_back = "in front" if y > 0 else "behind"
        left_right = "to the right" if x < 0 else "to the left"
        return left_right, front_back
    elif dx == -1 and dy == 0:
        cardinal = "west"
        front_back = "in front" if x < 0 else "behind"
        left_right = "to the right" if y > 0 else "to the left"
        return front_back, left_right
    elif dx == 1 and dy == 0:
        cardinal = "east"
        front_back = "in front" if x > 0 else "behind"
        left_right = "to the left" if y < 0 else "to the right"
        return front_back, left_right
    else:
        return "Invalid direction vector"


def describe_object_x(o, state, relative=False):
    if relative:
        dx, dy = o.cur_pos[0] - state.agent_pos[0], o.cur_pos[1] - state.agent_pos[1]
        xd, yd = relative_position(state.agent_dir, (dx, dy))
        units = "rows" if xd in ["in front", "behind"] else "columns"
        return f"{abs(dx)} {units} {xd}"
    else:
        return f"column {o.cur_pos[0]}"


def describe_object_y(o, state, relative=False):
    if relative:
        dx, dy = o.cur_pos[0] - state.agent_pos[0], o.cur_pos[1] - state.agent_pos[1]
        xd, yd = relative_position(state.agent_dir, (dx, dy))
        units = "rows" if yd in ["in front", "behind"] else "columns"
        return f"{abs(dy)} {units} {yd}"
    else:
        return f"row {o.cur_pos[1]}"


def describe_object_state(o):
    if o.type == "bridge":
        return "intact" if o.is_intact else "damaged"

    if o.type == "door":
        if o.is_locked:
            return "locked"
        elif o.is_open:
            return "open"
        else:
            return "closed"

    return ""


def describe_object_color(o):
    if o.type in ["bridge", "hammer", "wall", "fireproof_shoes", "passage"]:
        return ""
    return o.color


def get_attribute(o, name):
    if name == "x":
        return o.cur_pos[0]
    if name == "y":
        return o.cur_pos[1]
    if name == "forward":
        return o.rel_forward
    if name == "turn":
        return o.rel_turn
    if name == "color":
        return describe_object_color(o)
    if name == "state":
        return describe_object_state(o)
    raise NotImplementedError("Attribute not supported!")


def is_identifiable(o, objects, attrs):
    for oo in objects:
        if o == oo:
            continue
        cnt = 0
        for a in attrs:
            cnt += get_attribute(o, a) == get_attribute(oo, a)
        if cnt == len(attrs):
            return False
    return True


def plural_step(n):
    return inflect.engine().plural("step", n)


def describe_object(o, objects, relative=True, partial=False, article=None):

    attrs = ["x", "y", "state", "color"]
    if partial:
        chosen_attrs = []
        for a in random.sample(attrs, len(attrs)):
            chosen_attrs.append(a)
            if is_identifiable(o, objects, chosen_attrs):  # and random.random() < 0.8:
                break
    else:
        chosen_attrs = attrs

    d = o.type
    if "color" in chosen_attrs:
        o_color = describe_object_color(o)
        if o_color != "":
            d = o_color + " " + d
    if "state" in chosen_attrs:
        o_state = describe_object_state(o)
        if o_state != "":
            d = o_state + " " + d

    if ("x" in chosen_attrs or "y" in chosen_attrs) and not relative:
        d += " at"
    if "x" in chosen_attrs:
        if relative:
            d += f" {o.rel_forward} {plural_step(o.rel_forward)} forward"
        else:
            d += f" column {o.cur_pos[0]}"
    if "x" in chosen_attrs and "y" in chosen_attrs:
        d += " and"
    if "y" in chosen_attrs:
        if relative:
            d += f" {abs(o.rel_turn)} {plural_step(abs(o.rel_turn))} {'right' if o.rel_turn > 0 else 'left'}"
        else:
            d += f" row {o.cur_pos[1]}"

    if article is not None:
        if article == "the":
            d = "the " + d
        else:
            d = inflect.engine().a(d)

    return d


def describe_position(pos, obs_shape, relative=True):
    if relative:
        rel_forward = obs_shape[1] - 1 - pos[1]
        rel_turn = pos[0] - (obs_shape[0] // 2)
        d = f"{rel_forward} {plural_step(rel_forward)} forward"
        d += f" and {abs(rel_turn)} {plural_step(abs(rel_turn))} {'right' if rel_turn > 0 else 'left'}"
        return d
    return f"column {pos[0]} and row {pos[1]}"


def describe_obstacle_type(o):
    if o.type == "lava":
        return "lava stream"
    elif o.type == "wall":
        return "wall"


def describe_obstacle_direction(dim, state):
    dir_vec = DIR_TO_VEC[state.agent_dir]
    if dir_vec[0] == 0:
        return "vertical" if dim == 0 else "horizontal"
    assert dir_vec[1] == 0
    return "vertical" if dim == 1 else "horizontal"


def get_start_obstacle(obstacles, state):
    dir_vec = DIR_TO_VEC[state.agent_dir]
    dx = dy = 1
    if dir_vec[0] != 0:
        dx = dir_vec[0]
    if dir_vec[1] != 0:
        dy = dir_vec[1]
    best = None
    for o in obstacles:
        cand = (o.cur_pos[0] * dx, o.cur_pos[1] * dy)
        if best is None or cand < best[1]:
            best = (o, cand)
    return best[0]


def describe_obstacle_range(dim, obstacles, l, state, env, relative=False):
    o = get_start_obstacle(obstacles, state)
    d = f"a {describe_obstacle_direction(dim, state)} {describe_obstacle_type(o)} of width {env.obstacle_thickness}"
    if l == 1:
        d += " at"
    else:
        d += " " + f"and length {l} starting from"
    d += " " + f"{describe_object_x(o, state, relative=relative)}"
    d += " and " + f"{describe_object_y(o, state, relative=relative)}"
    return d


def describe_obstacle(o_type):
    if o_type == "lava":
        return "lava pool"
    if o_type == "safe_lava":
        return "cool lava pool"
    if o_type == "wall":
        return "wall"


def extract_objects_from_observation(obs):
    objects = []
    carrying = None
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i][j][0] in [
                OBJECT_TO_IDX["unseen"],
                OBJECT_TO_IDX["empty"],
                OBJECT_TO_IDX["wall"],
                OBJECT_TO_IDX["lava"],
                OBJECT_TO_IDX["safe_lava"],
            ]:
                continue

            o = SimpleNamespace(
                rel_forward=obs.shape[1] - 1 - j,
                rel_turn=i - (obs.shape[0] // 2),
                type=IDX_TO_OBJECT[obs[i][j][0]],
                color=IDX_TO_COLOR[obs[i][j][1]],
            )

            state = obs[i][j][2]
            if o.type == "bridge":
                o.is_intact = state
            elif o.type == "door":
                o.is_open = o.is_locked = 0
                if state == 0:
                    o.is_open = 1
                elif state == 2:
                    o.is_locked = 1

            if i == obs.shape[0] // 2 and j == obs.shape[1] - 1:
                carrying = o
            else:
                objects.append(o)
    return objects, carrying


def find_rectangles(grid, target_value):
    # Dimensions of the grid
    rows, cols = grid.shape
    # To keep track of whether a cell is already included in a rectangle
    included = np.zeros_like(grid, dtype=bool)
    rectangles = []

    # Iterate over each cell in the grid
    for r in range(rows):
        for c in range(cols):
            # Check if current cell is the target value and not already included
            if grid[r, c] == target_value and not included[r, c]:
                # Start a new rectangle
                start_r, start_c = r, c
                end_r, end_c = r, c

                # Expand to the right as far as possible
                while (
                    end_c + 1 < cols
                    and grid[start_r, end_c + 1] == target_value
                    and not included[start_r, end_c + 1]
                ):
                    end_c += 1

                # Try to expand downwards for all columns in the new rectangle
                valid_expansion = True
                while valid_expansion and end_r + 1 < rows:
                    for cc in range(start_c, end_c + 1):
                        if (
                            grid[end_r + 1, cc] != target_value
                            or included[end_r + 1, cc]
                        ):
                            valid_expansion = False
                            break
                    if valid_expansion:
                        end_r += 1

                # Mark all cells in this rectangle as included
                for rr in range(start_r, end_r + 1):
                    for cc in range(start_c, end_c + 1):
                        included[rr, cc] = True

                # Save the rectangle
                rectangles.append(((start_r, start_c), (end_r, end_c)))

    return rectangles


def describe_state(state, relative=True):

    if relative:
        obs = state.partial_obs
        objects, carrying = extract_objects_from_observation(obs)
    else:
        obs = state.full_obs
        objects = []
        for o in state.objects:
            is_visible = True
            if o.cur_pos == (-1, -1):
                is_visible = False
            for oo in state.objects:
                if hasattr(oo, "contains") and oo.contains == o:
                    is_visible = False
                    break
            if is_visible:
                objects.append(o)
        carrying = state.carrying


    d = f"You are at column {state.agent_pos[0]} and row {state.agent_pos[1]} . "
    d += f"You are facing {IDX_TO_DIR[state.agent_dir]} "
    # describe carried object
    if carrying:
        color = describe_object_color(carrying)
        dd = ""
        if color != "":
            dd += " " + color
        dd += " " + carrying.type
        d += f"and are carrying {inflect.engine().a(dd)} . "
    else:
        d += "and your inventory is empty . "
    # describe objects within view
    object_descriptions = []
    if objects:
        od = ", ".join(
            [inflect.engine().a(describe_object(o, objects, relative=relative)) for o in objects]
        )
        no = len(objects)
        d += f"You see {no} {inflect.engine().plural('object', no)} : {od} . "
    else:
        d += "You do not see any objects . "
    # describe obstacles
    obstacle_to_description = defaultdict(list)
    for o_type in ["wall", "lava", "safe_lava"]:
        if OBJECT_TO_IDX[o_type] in obs:
            rects = find_rectangles(obs[..., 0], OBJECT_TO_IDX[o_type])
            for p1, p2 in rects:
                p1_d = describe_position(p1, obs.shape, relative=relative)
                p2_d = describe_position(p2, obs.shape, relative=relative)
                o_name = describe_obstacle(o_type)
                dd = f"from {p1_d} to {p2_d}"
                obstacle_to_description[o_name].append(dd)

    for o_name, v in obstacle_to_description.items():
        d += f"There are {inflect.engine().plural(o_name)} : "
        d += ", ".join(v) + " . "

    d = re.sub(r"\s+", " ", d).strip()


    #print(obs[:, :, 0])
    #print(d)

    return d


"""
def describe_obstacles(state, env, relative=False):

    def find_ranges(dim):
        line_to_object = defaultdict(list)
        for o in obstacles:
            line_to_object[o.init_pos[dim]].append(o)

        for k in line_to_object:
            v = line_to_object[k]
            update_ranges(v, dim)

    def update_ranges(v, dim):
        v = sorted(v, key=lambda x: x.init_pos[1 - dim])
        last_i = None
        prev_o = None
        for i, o in enumerate(v):
            if prev_o is None or o.init_pos[1 - dim] - 1 != prev_o.init_pos[1 - dim]:
                if last_i is not None:
                    r = {"dim": dim, "length": l, "obstacles": []}
                    for j in range(last_i, i):
                        r["obstacles"].append(v[j])
                    for j in range(last_i, i):
                        object_to_range[v[j]].append(r)
                last_i = i
                l = 1
            else:
                l += 1
            prev_o = o

        r = {"dim": dim, "length": l, "obstacles": []}
        for j in range(last_i, len(v)):
            r["obstacles"].append(v[j])
        for j in range(last_i, len(v)):
            object_to_range[v[j]].append(r)

    obstacles = dc(env.obstacles)
    for o in state.objects:
        if isinstance(o, env.obstacle_cls):
            obstacles.append(dc(o))

    for o in obstacles:
        o.cur_pos = o.init_pos

    object_to_range = defaultdict(list)
    find_ranges(0)
    find_ranges(1)

    descriptions = []

    while True:
        can_break = True
        deleted_obstacles = []
        for k, v in object_to_range.items():
            max_l = max([x["length"] for x in v])
            for x in random.sample(v, len(v)):
                if x["length"] > 1 or (x["length"] == 1 and max_l == 1):
                    can_break = False
                    descriptions.append(
                        describe_obstacle_range(
                            x["dim"],
                            x["obstacles"],
                            x["length"],
                            state,
                            env,
                            relative=relative,
                        )
                    )
                    deleted_obstacles = x["obstacles"]
                    break
            if deleted_obstacles:
                break
        for o in deleted_obstacles:
            if o in object_to_range:
                del object_to_range[o]
        if can_break:
            break

    for d in descriptions:
        print(d)


def find_min_max_coords(points):
    if not points:
        return None  # Return None or raise an exception if the list is empty

    # Initialize min and max values with the coordinates of the first point
    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]

    # Iterate through the list of points and update min and max values
    for x, y in points:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, max_x, min_y, max_y


def describe_observation(state, env):

    d = "You are facing {IDX_TO_DIR[state.agent_dir]}."

    # describe section
    obstacle_d = []
    section_name = "a room" if env.layout_name == "room_door_key" else "an island"
    row_obstacles = {}
    col_obstacles = {}
    for o in range(env.obstacles):
        dx, dy = o.cur_pos[0] - state.agent_pos[0], o.cur_pos[1] - state.agent_pos[1]
        xd, yd = relative_position(DIR_TO_VEC[state.agent_dir], (dx, dy))
        col_obstacles[dx].append((dx, dy, xd, yd))
        row_obstacles[dy].append((dx, dy, xd, yd))

    for k, v in col_obstacles.items():
        v = sorted(v, key=lambda x: x[1])
        length = 1
        for i, e in enumerate(v):
            dx, dy, xd, yd = e
            if i > 0 and dy - 1 != e[i - 1][1]:
                obstacle_d.append(
                    f"There is a {env.obstacles[0].type} of length {width} {dx} cells {xd}"
                )
                length = 1
            length += 1
        obstacle_d.append(
            f"There is a {env.obstacles[0].type} of length {width} {dx} cells {xd}"
        )

    for o in state.objects:
        if o.type == "bridge":
            return
"""
