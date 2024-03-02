from package.enums import Level, Variant

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLORS, COLOR_NAMES, COLOR_TO_IDX, IDX_TO_COLOR, STATE_TO_IDX, DIR_TO_VEC

import numpy as np


OBJECT_TO_IDX["heavy_door"] = 11
OBJECT_TO_IDX["bridge"] = 12
OBJECT_TO_IDX["fireproof_shoes"] = 13
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

COLORS["brown"] = np.array([119, 70, 20])
COLOR_NAMES = sorted(list(COLORS.keys()))
COLOR_TO_IDX["brown"] = 6
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
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
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
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
    10: "Tenth"
}

PLAYABLE_OBJS = [Goal, Box, Ball]
TANGIBLE_OBJS = [Ball, Box]
DISTRACTOR_OBJS = [Box, Ball, Key]

OBJ_NAME_MAPPING = {
    Goal: "goal",
    Ball: "ball",
    Box: "box",
    Key: "key",
    Door: "door",
    Wall: "wall",
    Lava: "lava",
    # ...
}
NAME_OBJ_MAPPING = {v: k for k, v in OBJ_NAME_MAPPING.items()}
OBJ_PLURAL_MAPPING = {
    Goal: "goals",
    Ball: "balls",
    Box: "boxes",
    Key: "keys",
    # ...
}

AGENT_VIEW_SIZE = 5
MIN_VIEW_SIZE = 3
MAX_VIEW_SIZE = 9
MIN_ROOM_SIZE = 7
MAX_ROOM_SIZE = 12
MAX_NUM_LOCKED_DOORS = 3
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