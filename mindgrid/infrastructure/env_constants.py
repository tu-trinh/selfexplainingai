import numpy as np

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
    COLORS,
    COLOR_NAMES,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    STATE_TO_IDX,
    DIR_TO_VEC,
)


VEC_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_TO_NAME = {0: "right", 1: "down", 2: "left", 3: "up"}

OBJECT_TO_IDX["heavy_door"] = 11
OBJECT_TO_IDX["bridge"] = 12
OBJECT_TO_IDX["fireproof_shoes"] = 13
OBJECT_TO_IDX["hammer"] = 14
OBJECT_TO_IDX["passage"] = 15
OBJECT_TO_IDX["safe_lava"] = 16

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

COLORS["brown"] = np.array([119, 70, 20])
COLOR_NAMES = sorted(list(COLORS.keys()))
COLOR_TO_IDX["brown"] = 6
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
IDX_TO_DIR = {0: "east", 1: "south", 2: "west", 3: "north"}
ACTION_TO_IDX = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
CUSTOM_ACTION_TO_TRUE_ACTION = {1: 2, 2: 0, 3: 1, 4: 3, 5: 4, 6: 5, 7: 5, 8: 5}
SKILL_PHRASES = {
    "left": [lambda p: "left", lambda p: "turn left", lambda p: "make a left turn"],
    "right": [lambda p: "right", lambda p: "turn right", lambda p: "make a right turn"],
    "forward": [lambda p: "forward", lambda p: "step forward", lambda p: "go forward"],
    "backward": [
        lambda p: "backward",
        lambda p: "turn backward",
        lambda p: "make a U-turn",
    ],
    "pickup": [lambda p: "pickup", lambda p: "grab", lambda p: "snatch up"],
    "drop": [lambda p: "drop", lambda p: "lay down", lambda p: "set down"],
    "toggle": [lambda p: "toggle", lambda p: "switch", lambda p: "activate"],
    "move_forward": [
        lambda p: f"move_{p.split('_')[2]}_steps_forward",
        lambda p: f"advance_{p.split('_')[2]}_steps",
        lambda p: f"walk_{p.split('_')[2]}_cells_forward",
        lambda p: f"step_forward_{p.split('_')[2]}_times",
        lambda p: f"progress_{p.split('_')[2]}_steps_forward",
        lambda p: f"proceed_{p.split('_')[2]}_steps",
    ],
    "move_right": [
        lambda p: f"move_{p.split('_')[2]}_steps_right",
        lambda p: f"go_right_for_{p.split('_')[2]}_steps",
        lambda p: f"advance_{p.split('_')[2]}_steps_rightward",
        lambda p: f"walk_{p.split('_')[2]}_cells_right",
        lambda p: f"progress_{p.split('_')[2]}_steps_right",
        lambda p: f"proceed_right_{p.split('_')[2]}_steps",
    ],
    "move_left": [
        lambda p: f"move_{p.split('_')[2]}_steps_left",
        lambda p: f"go_left_for_{p.split('_')[2]}_steps",
        lambda p: f"advance_{p.split('_')[2]}_steps_leftward",
        lambda p: f"walk_{p.split('_')[2]}_cells_left",
        lambda p: f"progress_{p.split('_')[2]}_steps_left",
        lambda p: f"proceed_left_{p.split('_')[2]}_steps",
    ],
    "move_backward": [
        lambda p: f"move_{p.split('_')[2]}_steps_backward",
        lambda p: f"backpedal_{p.split('_')[2]}_steps",
        lambda p: f"go_back_{p.split('_')[2]}_steps",
        lambda p: f"retreat_{p.split('_')[2]}_paces",
        lambda p: f"proceed_backward_{p.split('_')[2]}_steps",
        lambda p: f"walk_{p.split('_')[2]}_cells_backward",
    ],
    "go_": [
        lambda p: f"go_to_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"proceed_towards_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"head_towards_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"approach_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"make_way_towards_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"navigate_to_{p.split('_')[-2]}_{p.split('_')[-1]}",
    ],
    "pickup_": [
        lambda p: f"pickup_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"acquire_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"retrieve_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"obtain_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"fetch_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"procure_{p.split('_')[-2]}_{p.split('_')[-1]}",
    ],
    "put_": [
        lambda p: f"put_down_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"put_{p.split('_')[-2]}_{p.split('_')[-1]}_away",
        lambda p: f"set_down_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"place_{p.split('_')[-2]}_{p.split('_')[-1]}_down",
        lambda p: f"put_{p.split('_')[-2]}_{p.split('_')[-1]}_on_floor",
        lambda p: f"lay_{p.split('_')[-2]}_{p.split('_')[-1]}_down",
    ],
    "open_": [
        lambda p: f"open_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"open_up_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"swing_open_{p.split('_')[-2]}_{p.split('_')[-1]}",
    ],
    "close_": [
        lambda p: f"close_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"shut_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"close_up_{p.split('_')[-2]}_{p.split('_')[-1]}",
    ],
    "unlock_": [
        lambda p: f"unlock_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"unlatch_{p.split('_')[-2]}_{p.split('_')[-1]}",
        lambda p: f"unbolt_{p.split('_')[-2]}_{p.split('_')[-1]}",
    ],
}

NUM_TO_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
NUM_TO_ORDERING = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth",
    6: "Sxith",
    7: "Seventh",
    8: "Eighth",
    9: "Ninth",
    10: "Tenth",
}


AGENT_VIEW_SIZE = 5
MIN_VIEW_SIZE = 3
MAX_VIEW_SIZE = 9
MIN_ROOM_SIZE = 7
MAX_ROOM_SIZE = 12
MAX_NUM_LOCKED_DOORS = 3

"""
UNIVERSAL_VARIANTS = [Variant.COLOR, Variant.ROOM_SIZE, Variant.ORIENTATION]
ALLOWABLE_VARIANTS = {
    Level.EMPTY: UNIVERSAL_VARIANTS,
    Level.DEATH: UNIVERSAL_VARIANTS,
    Level.DIST: UNIVERSAL_VARIANTS + [Variant.NUM_OBJECTS, Variant.OBJECTS],
    Level.OPEN_DOOR: UNIVERSAL_VARIANTS,
    Level.BLOCKED_DOOR: UNIVERSAL_VARIANTS,
    Level.UNLOCK_DOOR: UNIVERSAL_VARIANTS,
    Level.HIDDEN_KEY: UNIVERSAL_VARIANTS,
    Level.GO_AROUND: UNIVERSAL_VARIANTS,
    Level.MULT_ROOMS: UNIVERSAL_VARIANTS + [Variant.NUM_ROOMS],
    Level.BOSS: UNIVERSAL_VARIANTS + [Variant.NUM_OBJECTS, Variant.OBJECTS, Variant.NUM_ROOMS]
}
"""
