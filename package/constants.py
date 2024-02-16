from package.enums import *

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box


#################
#  Access Keys  #
#################
OPENAI_KEY = "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"  # BERKELEY ONE
# OPENAI_KEY = "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"  # GMAIL ONE
SCALE_KEY = "cll08l8da0af41asy4ukcbhpn"
HUGGINGFACE_KEY = "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"


##################
#  LLM Querying  #
##################
RANDOM_SEED = 420
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 100
NUM_ACTIONS_GENERATED = 1
PROMPT_HISTORY_LIMIT = 3

MAX_MSG_TOKENS = 3800
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4097
}

MAX_TRIES = 2


#######################
#  Environment Setup  #
#######################
PLAYABLE_OBJS = [Goal, Box, Ball]
TANGIBLE_OBJS = [Ball, Box]
DISTRACTOR_OBJS = [Box, Ball, Key]
DOOR_KEY_COLOR_NAMES = ["blue", "purple", "yellow", "red", "green", "grey"]
OBJECT_COLOR_NAMES = ["red", "green", "blue", "purple", "yellow"]
ALL_COLOR_NAMES = OBJECT_COLOR_NAMES + ["grey"]

OBJ_NAME_MAPPING = {
    Goal: "goal",
    Ball: "ball",
    Box: "box",
    Key: "key",
    Door: "door",
    Wall: "wall",
    Lava: "lava"
}
OBJ_PLURAL_MAPPING = {
    Goal: "goals",
    Ball: "balls",
    Box: "boxes",
    Key: "keys"
}

AGENT_VIEW_SIZE = 5
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


#########################
#  Prompting Templates  #
#########################
# TODO: physics "reminders": can only unlock door with same key, when you open a box it will disappear, can only hold one thing at a time (must drop before picking up something else), pickup and putting down require empty cell in front??
SETTING_DESCRIPTION = """You are an agent who is trying to complete a task in an unknown, grid-like environment. Keep in mind that in this environment, you can only unlock a locked door with a key of the same color, you can only carry one object at a time, and you can only put an object down in a cell that has no other object.
"""
"""
Here is an example trajectory of another agent like you who had the task of picking up a ball:
Observation 1: You are facing right. Your field of vision is a 3x3 square in which you are located at the bottom middle. You see walls and a closed blue door. Directly in front of you is the closed blue door.
Chosen action 1: 10. Open the blue door.
Observation 2: You are facing right. Your field of vision is a 3x3 square in which you are located at the bottom middle. You see walls.
Chosen action 2: 1. Explore what is around.
Observation 3: From your exploration, you know that you are facing right. Around you, you see the following: a wall 1 cell left and 1 cell behind, a wall 1 cell behind, a wall 1 cell right and 1 cell behind, a wall 1 cell left, a wall 1 cell right, a wall 1 cell left and 1 cell in front, a wall 1 cell right and 1 cell in front, a wall 1 cell left and 2 cells in front, and a wall 1 cell right and 2 cells in front.
Chosen action 3: 2. Go to the cell in front.
Observation 4: You are facing right. Your field of vision is a 3x3 square in which you are located at the bottom middle. You see walls and a yellow ball.
Chosen action 4: 6. Go to the yellow ball.
Observation 5: You are facing right. Your field of vision is a 3x3 square in which you are located at the bottom middle. You see walls and a yellow ball. Directly in front of you is the yellow ball.
Chosen action 5: 7. Pick up the yellow ball.
[SUCCESS]
"""
PROMPT_FORMAT_INSTRUCTION = """Below are previous observations you've seen in trying to achieve your task and your chosen action for each.\n"""
INQUIRY = """\nWhat action do you want to take in this timestep? Choose from the following list.
1. Explore what is around.
2. Go to the cell in front.
3. Go to the cell to the left.
4. Go to the cell to the right.
5. Go to the cell behind.
6. Go to the [COLOR] [OBJECT].
7. Pick up the [COLOR] [OBJECT].
8. Put down the [COLOR] [OBJECT].
9. Unlock the [COLOR] [OBJECT].
10. Open the [COLOR] [OBJECT].
11. Close the [COLOR] [OBJECT].
Note that only a subset of these actions will be valid given the observation you're currently seeing; use your best judgment. Your response should ONLY be EXACTLY ONE of the choices, such as \"1. Explore what is around.\", or \"6. Go to the blue box.\" (note that you must fill in the [COLOR] and [OBJECT] brackets in such cases). If none of the actions seem feasible, say \"I'm stumped.\""""
# INQUIRY = """\nWhat action do you want to take in this timestep? Choose ONE ACTION from the list you are given. Format your response concisely, EXACTLY like in the example trajectory above; for example, if you want to explore, say \"1. Explore what is around.\" If you see a blue box and you want to go to it, say \"6. Go to the blue box.\" (note that you must fill in the \"[COLOR]\" and \"[OBJECT]\" brackets in such cases) If none of the actions seem feasible, say, \"I'm stumped.\""""
TASK_PROLOGUE = "\nYOUR TASK IS: "
INSTRUCTION_PROLOGUE = " To do so, follow these steps exactly: "

DIFFERENCES_MAPPING = {
    "A": "Environment two is smaller",
    "B": "Environment two is bigger",
    "C": "Environment two is environment one rotated 90 degrees to the right",
    "D": "Environment two is environment one rotated 180 degrees to the right",
    "E": "Environment two is environment one rotated 270 degrees to the right",
    "F": "Environment two has fewer objects than environment one",
    "G": "Environment two has more objects than environment one",
    "H": "Environment two has the same number of objects as environment one, but the objects are different",
    "I": "Environment two has fewer doors than environment one",
    "J": "Environment two has more doors than environment one",
    "K": "Environment two has the same number of doors as environment one, but they are in different states (i.e. locked, open, or closed)",
    "L": "Environment two has fewer rooms than environment one",
    "M": "Environment two has more rooms than environment one"
}
DIFFERENCES_QUESTION = """Here is a description environment one: {other_env_desc}
Here is a description of environment two: {own_env_desc}
Which of the following reflects the differences in these two environments? There can be multiple answers.
{differences}
Please give only the letter(s) in your response in a comma-separated list, such as 'A' or 'A,D,E'."""

GET_SKILL_NAME_QUESTION = """Below is a sequence of observations and corresponding actions taken by an AI agent in trying to execute a 'skill'. Observations should be interpreted as environment descriptions told to the agent. Given this, please assign a name or short description to the skill that the agent is doing. Your response should just be the word or phrase for the skill: for example, 'pick up key', 'turn around', 'unlock door', etc. Don't say anything else.
Sequence of observations (obs) and actions (act):
{obs_act_seq}
"""