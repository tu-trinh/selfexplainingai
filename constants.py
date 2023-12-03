from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
# TODO: physics "reminders": can only unlock door with same key, when you open a box it will disappear, can only hold one thing at a time (must drop before picking up something else)

OPENAI_KEY = "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"  # BERKELEY ONE
# OPENAI_KEY = "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"  # GMAIL ONE
RANDOM_SEED = 420
TEMPERATURE = 0.0
NUM_ACTIONS_GENERATED = 1

ORIGINAL_SETTING_DESCRIPTION = """You are an agent who is trying to complete a task in an unknown, grid-like room. At each step, you may choose to execute one of the following actions:
1. Turn left. For example, if you are in cell (0, 0) and facing north, turning left would have you in the same cell but facing west.
2. Turn right. For example, if you are in cell (0, 0) and facing north, turning right would have you in the same cell but facing east.
3. Move forward. For example, if you are in cell (0, 0) and facing north, moving forward would put you in (0, 1).
4. Pick up an object. The object must be in a cell cardinally adjacent to you and you must be directly facing it. For example, if you are in cell (0, 0) and facing east, and there is a ball at (1, 0), you can pick up the ball and will be able to travel with it until you put it down.
5. Drop an object. There must be space directly in front of you to drop it. For example, if you are in cell (0, 0), facing north, and holding a key, and there is nothing at (0, 1), you can put the key down on (0, 1).
6. Unlock, open, or close a door. The door must be in a cell cardinally adjacent to you and you must be directly facing it. If it is locked, you must be holding a key of the same color as the door. For example, if you are in cell (0, 0), facing north, and holding a green key, and there is a green door at (0, 1), you can use the key to unlock the door, after which the door will also open."""

ORIGINAL_INQUIRY = """What is the next action you take in order to complete your given task? Please tell me the number corresponding to your chosen action. If you choose 4 (pick up) or 5 (drop), please also tell me the object you'll be interacting with. If none of the actions seem feasible, say, 'I'm stumped.'"""

SETTING_DESCRIPTION = """You are an agent who is trying to complete a task in an unknown, grid-like environment. At each step, you may choose to do one of the following actions:
1. Explore what is around you.
2. Go to the cell in front.
3. Go to the cell to the left.
4. Go to the cell to the right.
5. Go to the cell behind you.
"""

INQUIRY = """ What is the next action you take? Choose from the list you were given. If none of the actions seem feasible, say, 'I'm stumped.'"""

GOTO_TARGET_OBJS = [Goal, Ball, Box, Key]
PICKUP_TARGET_OBJS = [Ball, Box, Key]
DISTRACTOR_OBJS = [Ball, Box, Key]
DOOR_KEY_COLOR_NAMES = ["blue", "purple", "yellow", "red", "green"]
OBJECT_COLOR_NAMES = ["red", "green", "blue", "purple", "yellow"]
OBJ_NAME_MAPPING = {
    Goal: "goal",
    Ball: "ball",
    Box: "box",
    Key: "key"
}

AGENT_VIEW_SIZE = 5
MIN_ROOM_SIZE = 7
MAX_ROOM_SIZE = 12

MAX_TRIES = 2

MAX_MSG_TOKENS = 3800
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4097
}
