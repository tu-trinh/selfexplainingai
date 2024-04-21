from package.infrastructure.env_constants import MAX_ROOM_SIZE, COLOR_NAMES
from package.infrastructure.obj_constants import OBJ_NAME_MAPPING


TEMPERATURE = 0.01
MAX_NEW_TOKENS = 150
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
TASK_PROLOGUE = "\nYOUR TASK IS: "
INSTRUCTION_PROLOGUE = " To do so, follow these steps exactly: "
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
Below is a sequence of observations and corresponding actions taken by an AI agent in trying to execute a 'skill.' Observations should be interpreted as environment descriptions told to the agent. In this environment, names of skills are always succinct, specify object color when appropriate, and have underscores: for example, 'move_forward_2_steps', 'put_blue_key_on_floor', 'unlatch_green_door', or 'procure_brown_box'. Given this sequence below, what is the skill name that best describes the agent's trajectory? Do NOT say anything else except your chosen skill name.

Sequence of observations (obs) and actions (act):
{obs_act_seq}
"""
# """
# Important notes about the skill choices you might see:
# 1. Special meanings for single-word directional skills: 'left' means ONLY the act of turning left (not MOVING left). Likewise, 'right' means ONLY turning right. 'backward' means ONLY turning backwards. 'forward', on the other hand, means to take ONE step forward into the cell in front.
# 2. Directional skills that are not single-word should be self-explanatory; for example, move_left_1_steps means to turn left and then take one step forward.
# 2. Special meanings for single-world interaction skills: 'pickup' denotes ONLY the act of physically picking up whatever is in front. Likewise, 'drop' means ONLY the act of physically putting down whatever the agent is holding. 'toggle' means ONLY the act of physically toggling the object in front. It could be physically opening it, closing it, etc. depending on what the object is.
# 4. Interaction skills that are not single-word describe a longer sequence that terminate with the corresponding single-word skill. For example, 'pickup_green_ball' denotes the act of first navigating to the green ball such that the agent is facing it head-on, and then physically picking it up.

# IMPORTANT: If the agent only executes one action in the sequence, your answer MUST be the EXACT action string that follows 'Act 1'. Otherwise, if and only if the agent executes more than one action, choose the highest-level matching skill name possible.
# """

GET_NEW_PLAN_BASED_ON_SKILLS_QUESTION = """
Below you will be given a sequence of observations and corresponding actions taken by agent 1 in executing the following task: {task}. The observations should be interpreted as environment descriptions told to agent 1. There is a second agent, agent 2, who has a (potentially) different set of skills compared to agent 1 but who also wants to achieve the same task. Given agent 1's trajectory and agent 2's skills, please come up with a skill sequence for agent 2 that will help it achieve the same task in as few steps as possible. Your answer should just be comma-separated skills chosen from the list; don't say anything else.

Agent 1 sequence of observations (obs) and actions (act):
{obs_act_seq}

Agent 2 skills:
{skill_choices}
"""

SKILL_CLASSIFICATION = {
    0: ["left", "right", "forward", "pickup", "drop", "toggle"],
    1: [f"move_{dir}_{n}_steps" for dir in ["left", "right", "forward", "backward"] for n in range(1, MAX_ROOM_SIZE - 2)] + ["backward"],
    2: [f"go_to_{color}_{obj}" for color in COLOR_NAMES for obj in OBJ_NAME_MAPPING.values()],
    3: [f"pickup_{color}_{obj}" for color in COLOR_NAMES for obj in OBJ_NAME_MAPPING.values()] + [f"open_{color}_{obj}" for color in COLOR_NAMES for obj in ["door", "box"]] + [f"unlock_{color}_door" for color in COLOR_NAMES] + [f"close_{color}_door" for color in COLOR_NAMES]
}
BUILD_SKILL_TREE_PROMPT = """
Below you will be given a sequence of observations and corresponding actions ('obs-act sequence') taken by an AI agent in executing this task: {task}. Please help me group this sequence of actions into increasingly higher levels of abstract actions. Action names and the levels they belong to are here:
{skill_classification}

You will see that the actions provided in the obs-act sequence are all at the lowest level of abstraction. Some of them can be grouped into actions at the next higher level. Here are some examples of what I mean:
- Three 'forward' actions (level 0) can be grouped into a 'move_forward_3_steps' action (level 1).
- If the agent finds itself facing a blue ball after taking a 'move_forward_3_steps' action and a 'move_left_4_steps' action (both level 1), then they can both be grouped into a `go_to_blue_ball` action (level 2).
- If the agent goes towards a yellow box and then picks it up, this can be grouped into the 'pickup_yellow_box' action (level 3), which directly consists of a 'go_to_yellow_box' action (level 2) and a 'pickup' action (level 0).
You can start to see how these actions can be grouped hierarchically together in a tree-like structure.

Given the obs-act sequence below, please respond with the grouped actions in nested JSON format, where each key is a number indicating the action number/order and each value is a dictionary with two fields: 'name', indicating the action name, and 'children', a list of similarly-formatted JSONs of the constituent lower level actions. If the action is already level 0, this list should be empty. Do not mix up action numbering between abstraction levels—each abstraction level has its own action counter. As an example, here is how you might format your response if the agent has the simple task of picking up a yellow box (using my previous example):
{
	"0":{
		"name":"pickup_yellow_box",
		"children":[
			{
				"0":{
					"name":"go_to_yellow_box",
					"children":[
						{
							"0":{
								"name":"move_right_2_steps",
								"children":[
									{
										"0":{
											"name":"right",
											"children":[]
										},
										"1":{
											"name":"right",
											"children":[]
										}
									}
								]
							}
						}
					]
				},
				"1":{
					"name":"pickup",
					"children":[]
				}
			}
		]
	}
}

Now it is your turn! Here is the sequence of observations (obs) and actions (act). (Observations should be interpreted as environment descriptions told to the agent.):
{obs_act_seq}
"""