from package.infrastructure.env_constants import DIR_TO_VEC, MAX_ROOM_SIZE, COLOR_NAMES, OBJ_NAME_MAPPING

from typing import Tuple
from gymnasium import Env
import copy


"""
(Extended) Primitive Minigrid Actions
"""
def left():
    return [0]

def right():
    return [1]

def forward():
    return [2]

def pickup():
    return [3]

def drop():
    return [4]

def toggle():
    return [5]


"""
Basic directional movement
"""
def backward():
    return [0, 0]

def move_direction_n_steps_hof(direction: str, n: int):
    assert direction in ["left", "right", "forward", "backward"]
    assert 1 <= n <= MAX_ROOM_SIZE - 3

    def move_direction_n_steps():
        actions = []
        if direction == "left":
            actions.append(0)
        elif direction == "right":
            actions.append(1)
        elif direction == "backward":
            actions.extend([0, 0])
        for _ in range(n):
            actions.append(2)
        return actions
    
    return move_direction_n_steps


"""
Going to an object
"""
def go_to_color_object_hof(color: str, obj: str):
    assert color in COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def go_to_color_object(env: Env, object_pos: Tuple[int, int]):
        actions = []
        actions.extend(_find_path(env, object_pos, "goto", can_overlap = obj == "goal"))
        return actions
    
    return go_to_color_object


"""
Picking up an object
"""
def pickup_color_object_hof(color: str, obj: str):
    assert color in COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def pickup_color_object(env: Env, object_pos: Tuple[int, int]):
        actions = []
        actions.extend(_find_path(env, object_pos, action_type = "pickup"))
        return actions
    
    return pickup_color_object


"""
Putting down an object
"""
def put_down_color_object_hof(color: str, obj: str):
    assert color in COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def put_down_color_object():
        actions = [4]
        return actions
    
    return put_down_color_object

"""
Opening an object
"""
def open_color_object_hof(color: str, obj: str):
    assert color in COLOR_NAMES
    assert obj in ["door", "box"]

    def open_color_object(env: Env, object_pos: Tuple[int, int]):
        actions = []
        actions.extend(_find_path(env, object_pos, "goto"))
        actions.append(5)
        return actions
    
    return open_color_object

"""
Unlocking a door
"""
def unlock_color_door_hof(color: str, necessary_key_pos: Tuple[int, int]):
    assert color in COLOR_NAMES

    def unlock_color_door(env: Env, door_pos: Tuple[int, int]):
        actions = []
        actions.extend(_find_path(env, necessary_key_pos, "pickup"))
        new_env = copy.deepcopy(env)
        for action in actions:
            new_env.step(action)
        actions.extend(_find_path(new_env, door_pos, "goto", reset = False))
        actions.append(5)
        return actions
    
    return unlock_color_door

"""
Closing a door
"""
def close_color_door_hof(color: str):
    assert color in COLOR_NAMES

    def close_color_door(env: Env, door_pos: Tuple[int, int]):
        actions = []
        actions.extend(_find_path(env, door_pos, "goto"))
        actions.append(5)
        return actions
    
    return close_color_door


# TODO? Even higher level skills?
# - Unblock door
# - Put on fireproof shoes
# ...?


"""
Helper skills
"""
def _find_path(master_env: Env, object_pos: Tuple[int, int], action_type: str, can_overlap: bool = False, reset: bool = True):
    env = copy.deepcopy(master_env)
    if reset:
        env.reset()
    if action_type == "goto" and can_overlap:
        def goal_check(state: State):
            return state.loc == object_pos
    elif action_type == "goto" and not can_overlap:
        def goal_check(state: State):
            dir_vec = DIR_TO_VEC[state.dir]
            in_front = state.grid.get(state.loc[0] + dir_vec[0], state.loc[1] + dir_vec[1])
            return manhattan_distance(state.loc, object_pos) == 1 and in_front == state.grid.get(*object_pos)
    elif action_type == "pickup":
        def goal_check(state: State):
            return manhattan_distance(state.loc, object_pos) == 1 and state.carrying == state.grid.get(*object_pos)
    search_problem = Search("bfs", env, goal_check, "s")
    actions = search_problem.search()
    if actions is None:  # TODO: idk
        return [0]
    return actions
