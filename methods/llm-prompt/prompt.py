import argparse
import sys
import os
import time
from tqdm import tqdm

sys.path.append(".")
import pickle


from mindgrid.envs.edits import Edits
from mindgrid.access_tokens import *
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.env_utils import describe_state
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.skills import Skills

import llmengine
from llmengine import Completion
import json
from typing import Dict

MODELS = ["llama-3-70b-instruct", "mixtral-8x7b-instruct", "gemma-7b-instruct"]
TEMPERATURE = 0.01
RDK_INTRO = """
You are an AI agent helping a human play a 2D grid-based game. The goal of the game is to {goal} on the grid. Here are the key rules of the game:
1. You can pick up objects like keys, balls, boxes, but your inventory can hold only one object at a time (a pair of shoes counts as one object).
2. You can unlock a locked door with a key that has the same color as the door.
3. You can only put an object down in a cell that doesn’t already contain another object.
4. When you open a box, it disappears and is replaced by whatever was inside it, if there was something.
"""
TI_INTRO = """
You are an AI agent helping a human play a 2D grid-based game. The goal of the game is to {goal} on the grid. Here are the key rules of the game:
1. You can pick up objects like keys, balls, boxes, hammers, and fireproof shoes, but your inventory can hold only one object at a time (a pair of shoes counts as one object).
2. If you step on lava, you die instantly unless the lava has been cooled or you are carrying fireproof shoes.
3. You can cross bridges safely unless they are damaged. Damaged bridges can be repaired with a hammer.
4. You can only put an object down in a cell that doesn’t already contain another object.
5. When you open a box, it disappears and is replaced by whatever was inside it, if there was something.
"""

ACTION_DESC = """
During each turn, you can perform one action. The available actions are:
1. forward: Move forward by one cell.
2. left: Turn 90 degrees to the left.
3. right: Turn 90 degrees to the right.
4. pickup: Pick up an object directly in front of you if it can be picked up.
5. drop: Drop the object you're holding into the cell directly in front of you.
6. toggle: Interact with an object in front of you (e.g., open a box or door).
"""


def show_env(env):
    env.render_mode = "human"
    env.render()
    input()


def make_example(task, datapoint, include_answer=False):
    config = make_config(config_str=datapoint["config"])
    true_agent_env = make_env(config.true_agent.env)
    false_agent_env = make_env(config.false_agent.env)

    true_agent_env.reset()
    false_agent_env.reset()

    #show_env(false_agent_env)
    #show_env(true_agent_env)

    true_agent_env_desc = describe_state(true_agent_env.get_state(), relative=False)
    prompt = f"What you observe on the grid: {true_agent_env_desc}" + "\n\n"
    prompt += "The human's plan:\n"

    plan = datapoint["ref_plan"]["false_agent"]
    t = Trajectory()
    for i, (s, a) in enumerate(plan):
        obs_desc = describe_state(false_agent_env.get_state(), relative=True)
        skill = to_enum(Skills, s).value(**a)
        act_desc = skill.verbalize(false_agent_env)
        t += skill(false_agent_env)
        # prompt += f"Step {i + 1}:\nObservation: {obs_desc}\nAction: {act_desc}\n"
        prompt += f"Step {i + 1}: {act_desc}\n"
    prompt += "\n"
    # obs_desc = describe_state(false_agent_env.get_state(), relative=True)
    # prompt += f"Final observation: {obs_desc}\n\n"

    prompt += "Answer: "
    if include_answer:
        descriptions = []
        edits = true_agent_env.applied_edits[len(false_agent_env.applied_edits) :]
        for e in edits:
            edit_desc = e.verbalize()
            edit_desc = edit_desc[0].upper() + edit_desc[1:] + "."
            descriptions.append(edit_desc)
        prompt += " ".join(descriptions) + "\n"

    return prompt


def build_prompt(
    datapoint: Dict,
    few_shot,
    train_data,
) -> str:
    config = make_config(config_str=datapoint["config"])
    true_agent_env = make_env(config.true_agent.env)
    false_agent_env = make_env(config.false_agent.env)

    true_agent_env.reset()
    false_agent_env.reset()

    # print(config)

    is_rdk = "room_door_key" in datapoint["config"]
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    prompt = prompt.format(goal=true_agent_env.mission)

    prompt += f"The human player proposed a plan to {true_agent_env.mission}. However, the plan was based on an outdated version of the grid. Since that time, several changes have been made to the grid. You will be provided with an observation of the current grid and the human's plan. The plan is guaranteed to achieve the desired goal on the old grid. Your task is to infer the changes made to the grid. These changes were made sequentially, so you must list them in the correct order. You MUST use the following sentence templates to describe the changes:\n"
    for i, edit in enumerate(Edits):
        prompt += f'{i + 1}. "{edit.value.get_template(true_agent_env)}"\n'
    prompt += "\n"
    prompt += """In these templates: {row} or {col} is a row or column index; {color} is a color name; {state} is a state of a door or a bridge (`closed`, `open`, or `locked` for door, and `damaged` or `intact` for bridge), {tool} is either `key` or `hammer`. Do not change words that are not enclosed in braces.\n\n"""

    prompt += "Your answer should be a paragraph in which each sentence is constructed from one of the templates. Do not output anything else. For example: The color of the target object has been changed to blue. There is a walkable passage at row 1 and column 5.\n\n"

    if few_shot:
        prompt += "Here are a few examples to familiarize you with this task:\n\n"
        for i in range(few_shot):
            prompt += f"<example>\n"
            prompt += make_example(task, train_data[i], include_answer=True)
            prompt += f"</example>\n\n"

        prompt += "Now, answer the following case:\n\n"

    prompt += make_example(task, datapoint, include_answer=False)

    #print(prompt)
    #input()
    # true_agent_env.reset()
    # false_agent_env.reset()
    # show_env(false_agent_env)
    # show_env(true_agent_env)

    return prompt


def load_data(version, prefix, split):
    with open(f"datasets/{prefix}_games_5000_v{version}.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"{split}"] if split != "train" else games["train"]
    return games


def count_lines(save_file):
    # Open the file in read mode
    with open(save_file, "r") as file:
        # Read the lines in the file
        lines = file.readlines()

    # Count the number of lines
    number_of_lines = len(lines)
    print("Number of lines in the file:", number_of_lines)
    return number_of_lines


def execute_plan(env, plan):
    env.reset()
    t = Trajectory()
    for skill, kwargs in plan:
        t += Skills[skill].value(**kwargs)(env)
        if t.is_null:
            t = Trajectory()
    success = (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )
    return success * 100 - env.step_count


def check_value_diff():
    cnt = 0
    for i, game in enumerate(games):
        # if i != 2:
        #    continue
        game_config = make_config(config_str=game["config"])
        false_agent_env = make_env(game_config.false_agent.env)
        true_agent_env = make_env(game_config.true_agent.env)

        # true_agent_env.render_mode = "human"
        # true_agent_env.reset()
        # true_agent_env.render()
        # input()

        true_r = execute_plan(true_agent_env, game["ref_plan"]["true_agent"])
        false_r = execute_plan(true_agent_env, game["ref_plan"]["false_agent"])

        cnt += true_r - false_r

        print(i, cnt, true_r, false_r)

    print(cnt / len(games))

    input()

def find_start_idx(result_file):
    data = []
    cnt = 0
    game_id = output = None
    with open(result_file, "r") as file:
        for line in file:
            if line.startswith("env-test_out"):
                if game_id is not None:
                    cnt += 1
                game_id, output = line.split("\t")
            else:
                output += line
    cnt += 1
    return cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_version", type=int, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--task", type=str, default="teach")
    parser.add_argument("--model_id", type=int, default=0)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--few_shot", type=int, default=0)

    args = parser.parse_args()

    llmengine.api_engine.api_key = SCALE_KEY

    version = args.version
    prefix = args.prefix
    task = args.task
    few_shot = args.few_shot
    model = MODELS[args.model_id]

    train_games = load_data(version, prefix, "train")
    test_games = load_data(version, prefix, "test_out")
    save_file = f"methods/llm-prompt/results/{prefix}_{task}_5000_v{version}.{few_shot}-shot.{model}.prompt-v{args.prompt_version}.out"

    print(f"Save to {save_file} ?")
    input()

    start_idx = 0
    if os.path.exists(save_file):
        start_idx = find_start_idx(save_file)

    print(f"Starting from {start_idx}")

    for i, game in enumerate(tqdm(test_games[: args.num_games])):
        if i >= start_idx:
            prompt = build_prompt(game, args.few_shot, train_games)

            resp = Completion.create(
                model=model,
                prompt=prompt,
                temperature=TEMPERATURE,
                max_new_tokens=250,
            )

            model_answer = json.loads(resp.json())["output"]["text"]
            # print(model_answer)
            # input()
            with open(save_file, "a") as f:
                f.write(game["id"] + "\t" + model_answer + "\n")

            time.sleep(2)
