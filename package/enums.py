from enum import Enum

class Task(Enum):
    PICKUP = "Pickup"  # include returning to original spot

    # current not supported!
    GOTO = "Goto" # go to a location
    PUT = "PUT" # put an object next to another object
    COLLECT = "Collect" # put all objects in a region
    CLUSTER = "Cluster"  # group objects by type or color

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


class Layout(Enum):

    ROOM_DOOR_KEY = "Room_Door_Key"
    TREASURE_ISLAND = "Treasure_Island"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


class Edit(Enum):

    # applicable to all environments
    NONE = "none"
    DOUBLE_GRID_SIZE = "double_grid_size"
    FLIP_VERTICAL = "flip_vertical"
    CHANGE_TARGET_COLOR = "change_target_color"
    HIDE_TARGETS_IN_BOXES = "hide_targets_in_boxes"
    CHANGE_AGENT_VIEW_SIZE = "change_agent_view_size"
    ADD_OPENING = "add_opening"
    TOGGLE_OPENING = "toggle_opening"
    ADD_PASSAGE = "add_passage"
    BLOCK_OPENING = "block_opening"
    PUT_AGENT_INSIDE_SECTION = "put_agent_inside_section"
    HIDE_TOOL_IN_BOX = "hide_tool_in_box"
    REMOVE_TOOL = "remove_tool"

    # treasure_island only
    MAKE_LAVA_SAFE = "make_lava_safe"
    ADD_FIREPROOF_SHOES = "add_fireproof_shoes"

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)


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

    #ROOM_DOOR_KEY = "Room_Door_Key"
    #TREASURE_ISLAND = "Treasure_Island"

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
