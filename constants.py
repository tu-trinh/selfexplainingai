"""
4: ... Trying to pick up an object that is not directly in front of you won't do anything.
6: ... Trying to unlock an unlocked door or opened door won't do anything. Trying to unlock a door that is not directly in front of you won't do anything.
7: ... Trying to open a locked door without the corresponding key won't do anything. Trying to open an open door won't do anything. Trying to open a door that is not directly in front of you won't do anything.
8: ... Trying to close a closed door won't do anything. Trying to close a door that is not directly in front of you won't do anything.
"""

OPENAI_KEY = "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"
RANDOM_SEED = 420
TEMPERATURE = 0.1
NUM_ACTIONS_GENERATED = 1
SETTING_DESCRIPTION = """You are an agent who is trying to complete a task in an unknown, grid-like room. Your abilities include the following:
1. Moving forward one cell. For example, if you are in cell (0, 0) and facing north, moving forward would put you in (0, 1).
2. Turning left. For example, if you are in cell (0, 0) and facing north, turning left would have you in the same cell but facing west.
3. Turning right For example, if you are in cell (0, 0) and facing north, turning right would have you in the same cell but facing east.
4. Picking up an object (if it is in a cell cardinally adjacent to you and you are facing it). For example, if you are in cell (0, 0) and facing east, and there is a ball at (1, 0), you can pick up the ball and will be able to travel with it until you put it down.
5. Putting down an object (if there is empty floor directly in front of you to put it down). For example, if you are in cell (0, 0), facing north, and holding a key, and there is nothing except the floor at (0, 1), you can put the key down on (0, 1).
6. Unlock a door (if it is in a cell cardinally adjacent to you and you are facing it, and if you have a key of the same color as the door). For example, if you are in cell (0, 0), facing north, and holding a green key, and there is a green door at (0, 1), you can use the key to unlock the door, after which the door will also open.
7. Open a door (if it is in a cell cardinally adjacent to you and you are facing it). For example, if you are in cell (0, 0) and facing east, and there is an unlocked door at (1, 0), you can open the door.
8. Close a door (if it is in a cell cardinally adjacent to you and you are facing it). For example, if you are in cell (0, 0) and facing east, and there is an open door at (1, 0), you can close the door.
Each of these abilities are primitive actions that must be performed step-by-step. For example, if you are in cell (0, 0) facing east and there is a closed door at (0, 3) which you want to go through, you cannot immediately perform an \"open door\" action. You must first \"move forward\" twice to (0, 2) so that the door is directly in front of you. Only then can you \"open door\"."""
INQUIRY = """What is the next action you take in order to complete your given task? Choose from the following list.
1. Move forward one cell.
2. Turn left.
3. Turn right.
4. Pick up [OBJECT]. (Object must be directly one cell in front of you!)
5. Put down [OBJECT]. (You must be currently holding the object and there must be empty floor directly one cell in front of you!)
6. Unlock door. (You must have a matching key and the door must be directly one cell in front of you!)
7. Open door. (Door must be closed/unlocked and directly one cell in front of you!)
8. Close door. (Door must be open and directly one cell in front of you!)
Adhere strictly to these response formatting rules: If you choose options 1-3 or 6-8, your response must be in the form \"[NUMBER]\"; for example, if you want to move forward, you must say \"1\". If you choose options 4 or 5, you must also specify the object you want to handle in the form \"[NUMBER], [OBJECT]\"; for example, to put down a key, you must say "5, key". Do not choose options 4 or 5 if the object you have in mind is not said to exist in the room. Do not choose options 6-8 if it is not said that a door exists in the room. If none of the options seem feasible, say, \"I'm stumped.\" Don't say anything else in your response that strays from these rules."""
AGENT_VIEW_SIZE = 5
DOOR_KEY_COLOR_NAMES = ["blue", "purple", "yellow", "red", "green"]
GOAL_COLOR_NAMES = ["green", "red", "yellow", "purple", "blue"]
MAX_TRIES = 2
MAX_MSG_TOKENS = 4080
MIN_ROOM_SIZE = 7
MAX_ROOM_SIZE = 12
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4097
}