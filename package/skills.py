from package.constants import *

"""
Primitive Minigrid Actions
"""
def turn_left():
    return [0]

def turn_right():
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
def move_DIRECTION_N_steps_hof(direction: str, n: int):
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
def go_to_COLOR_OBJECT_hof(color: str, obj: str):
    assert color in ALL_COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def go_to_color_object(agent_pos, agent_dir, object_pos):
        actions = []
        actions.extend(_find_path(agent_pos, agent_dir, object_pos, can_overlap = obj == "goal"))
        return actions
    
    return go_to_color_object


"""
Picking up an object
"""
def pickup_COLOR_OBJECT_hof(color: str, obj: str):
    assert color in OBJECT_COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def pickup_color_object(agent_pos, agent_dir, object_pos):
        actions = []
        actions.extend(_find_path(agent_pos, agent_dir, object_pos))
        actions.append(3)
        return actions
    
    return pickup_color_object


"""
Putting down an object
"""
def put_down_COLOR_OBJECT_hof(color: str, obj: str):
    assert color in OBJECT_COLOR_NAMES
    assert obj in OBJ_NAME_MAPPING.values()

    def put_down_color_object():
        actions = [4]
        return actions
    
    return put_down_color_object

"""
Opening an object
"""
def open_COLOR_OBJECT_hof(color: str, obj: str):
    assert color in OBJECT_COLOR_NAMES
    assert obj in ["door", "box"]

    def open_color_object(agent_pos, agent_dir, object_pos):
        actions = []
        actions.extend(_find_path(agent_pos, agent_dir, object_pos))
        actions.append(5)
        return actions
    
    return open_color_object

"""
Unlocking a door
"""
def unlock_COLOR_door_hof(color: str):
    assert color in OBJECT_COLOR_NAMES

    def unlock_color_door(agent_pos, agent_dir, door_pos):
        actions = []
        actions.extend(_find_path(agent_pos, agent_dir, door_pos))
        actions.append(5)
        return actions
    
    return unlock_color_door

"""
Closing a door
"""
def close_COLOR_door_hof(color: str):
    assert color in OBJECT_COLOR_NAMES

    def close_color_door(agent_pos, agent_dir, door_pos):
        actions = []
        actions.extend(_find_path(agent_pos, agent_dir, door_pos))
        actions.append(5)
        return actions
    
    return close_color_door


"""
Helper skills
"""
def _find_path(agent_pos, agent_dir, object_pos, can_overlap = False):
    actions = []
    dx = object_pos[0] - agent_pos[0]
    dy = object_pos[1] - agent_pos[1]
    if dx > 0:
        required_dir_x = 0
    elif dx < 0:
        required_dir_x = 2
    if dy > 0:
        required_dir_y = 1
    elif dy < 0:
        required_dir_y = 3

    while agent_dir < required_dir_x:
        actions.append(1)
        agent_dir += 1
    while agent_dir > required_dir_x:
        actions.append(0)
        agent_dir -= 1
    for _ in range(abs(dx)):
        actions.append(2)

    while agent_dir < required_dir_y:
        actions.append(1)
        agent_dir += 1
    while agent_dir > required_dir_y:
        actions.append(0)
        agent_dir -= 1
    for _ in range(abs(dy) if can_overlap else abs(dy) - 1):
        actions.append(2)
    return actions
