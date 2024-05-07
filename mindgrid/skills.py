from mindgrid.infrastructure.env_constants import DIR_TO_VEC, MAX_ROOM_SIZE, COLOR_NAMES
from mindgrid.infrastructure.obj_constants import OBJ_NAME_MAPPING
from mindgrid.search import Search, State
from mindgrid.infrastructure.basic_utils import manhattan_distance, debug, get_adjacent_cells
from mindgrid.envs.modifications import Bridge

from minigrid.core.world_object import Wall, Door, Lava

from typing import Tuple, List, Union
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
        actions.extend(_find_path(env, object_pos, "goto", can_overlap = obj in ["goal", "bridge"]))
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

    def put_down_color_object(env: Env, object_pos: Tuple[int, int]):
        # object_pos here is the door which should not have objects blocking it
        actions = []
        actions.extend(_find_path(env, object_pos, "putdown"))
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
        actions.extend(_find_path(new_env, door_pos, "goto"))
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
Helper methods
"""
def _find_path(master_env: Env, object_pos: Union[Tuple[int, int], Tuple[Tuple[int, int]]], action_type: str, forbidden_actions: List[int] = [], can_overlap: bool = False, reset: bool = False):
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
        obj_to_pick = env.grid.get(*object_pos)
        def goal_check(state: State):
            correct_distance_away = manhattan_distance(state.loc, object_pos) == 1
            carrying = state.carrying is not None and state.carrying.color == obj_to_pick.color and type(state.carrying) == type(obj_to_pick)
            return correct_distance_away and carrying
    elif action_type == "putdown":
        def goal_check(state: State):  # object_pos here is the door which should not have objects blocking it
            clear_pos, _ = _check_clear_pos(state.loc, object_pos, state.grid)
            return state.carrying is None and clear_pos
    search_problem = Search("bfs", env, goal_check, "s", forbidden_actions)
    actions = search_problem.search()
    if actions is None:
        print("Oh no! No path found :(")
        return [0]
    return actions


def _check_clear_pos(agent_pos: Tuple[int, int], obj_pos: Union[Tuple[int, int], Tuple[Tuple[int, int]]], grid, is_bridge = False):
    # obj_pos: 1-2 door/bridge/objs we don't want blocked
    # is_bridge means obj_pos is a single bridge
    def helper(to_be_clear_pos: Tuple[int, int]):
        dir_to_agent = (agent_pos[0] - to_be_clear_pos[0], agent_pos[1] - to_be_clear_pos[1])
        # Finding where the agent is in relation to the door/bridge/obj
        agent_dir_locs = [False, False, False, False]  # right, below, left, above
        if dir_to_agent[0] < 0:
            agent_dir_locs[2] = True
        elif dir_to_agent[0] > 0:
            agent_dir_locs[0] = True
        if dir_to_agent[1] < 0:
            agent_dir_locs[3] = True
        elif dir_to_agent[1] > 0:
            agent_dir_locs[1] = True
        clear_obj = True
        blocker_obj = None
        # Special case where agent is standing inside the doorway
        # Then there should not be an obj behind it since it would have been removed if agent has gone through door
        if agent_pos == to_be_clear_pos:
            # debug("INSIDE HERE YIPPEE")
            # debug("to be clear pos", to_be_clear_pos)
            adjacent_cells = get_adjacent_cells(to_be_clear_pos)
            for ac in adjacent_cells:
                # debug("adj pos", ac)
                adj_obj = grid.get(*ac)
                if is_bridge:
                    clear_condition = lambda ao: ao is None or type(ao) == Lava
                else:
                    # if it's a door, we only allow surrounding walls. if it's an object, it's ok to have an (open) door next to it
                    clear_condition = lambda ao: ao is None or type(ao) == Wall or (type(ao) == Door and ao.is_open)
                if not clear_condition(adj_obj):
                    clear_obj = False
                    blocker_obj = adj_obj
                    break
        else:
            for i, adl in enumerate(agent_dir_locs):
                if adl:
                    dir_vec = DIR_TO_VEC[i]
                    adj_obj = grid.get(to_be_clear_pos[0] + dir_vec[0], to_be_clear_pos[1] + dir_vec[1])
                    if is_bridge:
                        clear_condition = lambda ao: ao is None or type(ao) == Lava
                    else:
                        clear_condition = lambda ao: ao is None or type(ao) == Wall or (type(ao) == Door and ao.is_open)
                        # if it's a door, we only allow surrounding walls. if it's an object, it's ok to have an (open) door next to it
                    if not clear_condition(adj_obj):
                        clear_obj = False
                        blocker_obj = adj_obj
                        break
        return clear_obj, blocker_obj
    if isinstance(obj_pos[0], tuple):
        for tbcp in obj_pos:  # if multiple positions need clearing. ONLY used for goal checking, NOT to detect if there is a blocking object
            clear_obj, _ = helper(tbcp)
            if not clear_obj:
                return False, None
        return True, None
    else:  # if just one position needing to be clear
        return helper(obj_pos)
