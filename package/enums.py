from enum import Enum


class Level(Enum):
    """
    EMPTY: Completely empty room except for the necessary object(s) to complete mission
    DEATH: Presence of lava that will kill the agent if touched
    DIST: Presence of random distractors of all types and colors
    OPEN_DOOR: Must open an unlocked door at some point to complete mission
    BLOCKED_DOOR: Must move an object away from the door before opening it
    UNLOCK_DOOR: Must find a key to unlock and open a door at some point to complete mission
    HIDDEN_KEY: Must open a box to retrieve the key before using it to open a door
    GO_AROUND: Must go around a line of walls or some other blocking object at some point
    MULT_ROOMS: Multiple rooms with doors of various open/unlocked states
    BOSS: Combine MULT_ROOMS, DIST, DEATH, and UNLOCK_DOOR
    """
    EMPTY = "Empty"
    DEATH = "Death"
    DIST = "Dist"
    OPEN_DOOR = "Open_Door"
    BLOCKED_DOOR = "Blocked_Door"
    UNLOCK_DOOR = "Unlock_Door"
    HIDDEN_KEY = "Hidden_Key"
    GO_AROUND = "Go_Around"
    MULT_ROOMS = "Mult_Rooms"
    BOSS = "Boss"

    ROOM_DOOR_KEY = "Room_Door_Key"
    TREASURE_ISLAND = "Treasure_Island"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


class Task(Enum):
    GOTO = "Goto"
    PICKUP = "Pickup"  # include returning to original spot
    PUT = "Put"
    COLLECT = "Collect"
    CLUSTER = "Cluster"  # group objects by type or color

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


class Variant(Enum):
    COLOR = "color"  # different target object(s) color(s)
    ROOM_SIZE = "room_size"  # different room size
    NUM_OBJECTS = "num_objects"  # different number of objects
    OBJECTS = "objects"  # same positions but different objects in those positions
    NUM_ROOMS = "num_rooms"  # different number of rooms
    ORIENTATION = "orientation"  # rotated some degrees

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


class MessageType(Enum):
    BELIEF_START = "belief_start"
    INTENTION_START = "intention_start"
    REWARD_START = "reward_start"
    MODEL_DESC = "world_model_description"
    SKILL_DESC = "skill_description"
    LANGUAGE_PLAN = "language_plan"
