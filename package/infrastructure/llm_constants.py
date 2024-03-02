TEMPERATURE = 0.1
MAX_NEW_TOKENS = 100
NUM_ACTIONS_GENERATED = 1
PROMPT_HISTORY_LIMIT = 3

MAX_MSG_TOKENS = 3800
CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4097
}

MAX_TRIES = 2

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

GET_SKILL_NAME_QUESTION = """
Below is a sequence of observations and corresponding actions taken by an AI agent in trying to execute a 'skill'. Observations should be interpreted as environment descriptions told to the agent. Given this sequence, please choose from the available choices the skill name that best describes the agent's trajectory. Don't say anything else except your chosen skill name.
Sequence of observations (obs) and actions (act):
{obs_act_seq}

Skill choices:
{skill_choices}
"""

GET_NEW_PLAN_BASED_ON_SKILLS_QUESTION = """
Below is a sequence of observations and corresponding actions taken by AI agent 1 in executing the following task: {task}. The observations should be interpreted as environment descriptions told to agent 1. Now there is another AI agent, agent 2, who has a (potentially) different set of skills but who also wants to achieve the same task. Given agent 1's sequence and agent 2's skills, please come up with a sequence of skills agent 2 should execute that will help it achieve the task in as few steps as possible. Your answer should just be comma-separated skills chosen from the list; don't say anything else.
Agent 1 sequence of observations (obs) and actions (act):
{obs_act_seq}

Agent 2 skills:
{skill_choices}
"""