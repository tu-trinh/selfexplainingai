from enum import Enum
from skills import *
from reward_functions import *


class Level(Enum):
    """
    EMPTY: Completely empty room except for the necessary object(s) to complete mission
    DEATH: Presence of lava that will kill the agent if touched
    DIST: Presence of random distractors of all types and colors
    OPEN_DOOR: Must open an unlocked door at some point to complete mission
    UNLOCK_DOOR: Must find a key to unlock and open a door at some point to complete mission
    GO_AROUND: Must go around a line of walls or some other blocking object at some point
    MULT_ROOMS: Multiple rooms with doors of various locked/unlocked states
    BOSS: Combine MULT_ROOMS, DIST_SAME, DIST_DIFF, and BLOCKED_DOOR
    """
    EMPTY = "Empty"
    DEATH = "Death"
    DIST = "Dist"
    OPEN_DOOR = "OpenDoor"
    BLOCKED_DOOR = "BlockedDoor"
    UNLOCK_DOOR = "UnlockDoor"
    HIDDEN_KEY = "HiddenKey"
    GO_AROUND = "GoAround"
    MULT_ROOMS = "MultRooms"
    BOSS = "Boss"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.value for item in cls)


class EnvType(Enum):
    GOTO = "Goto"
    PICKUP = "Pickup"
    PUT = "Put"
    COLLECT = "Collect"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.value for item in cls)


class Variant(Enum):
    COLOR = "color"
    ROOM_SIZE = "room_size"
    NUM_OBJECTS = "num_objects"
    NUM_ROOMS = "num_rooms"
    VIEW_SIZE = "view_size"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.value for item in cls)


class PrimitiveAction(Enum):
    LEFT = 0