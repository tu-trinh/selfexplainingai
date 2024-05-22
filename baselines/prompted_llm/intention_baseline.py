import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mindgrid.skills import Skills
from mindgrid.skills.skills import Primitive
from package.infrastructure.access_tokens import *
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.env import MindGridEnvState
from mindgrid.infrastructure.env_utils import describe_state
from mindgrid.infrastructure.env_constants import ACTION_TO_IDX, IDX_TO_ACTION

import llmengine
from llmengine import Completion
import argparse
import pickle
import json
import re
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import time
import numpy as np


MODELS = ["llama-3-70b-instruct", "mixtral-8x7b-instruct", "gemma-7b-instruct"]
TEMPERATURE = 0.01
RDK_INTRO = """
You are an agent who is trying to complete a task in a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has four special rules you must keep in mind:
1. You can only unlock a locked door with a key of the same color as the door
2. You can pick up keys, balls, and boxes, but you can only be carrying one object at a time
3. You can only put an object down in a cell that has no other object
4. When you open a box, it disappears and is replaced by the object inside it, IF there is one
"""
TI_INTRO = """
You are an agent who is trying to complete a task in a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has five special rules you must keep in mind:
1. You can pick up keys, balls, boxes, hammers, and fireproof shoes, but you can only be carrying one object at a time (a pair of shoes counts as one object)
2. If you step in lava you die instantly, but if the lava is cooled or if you are carrying fireproof shoes you will be fine
3. You can safely cross bridges if they are not damaged; you can fix damaged bridges with a hammer
4. You can only put an object down in a cell that has no other object
5. When you open a box, it disappears and is replaced by the object inside it, IF there is one
"""
TRAINING_DATA = None
NUM_TRAINING_EXAMPLES = 0
USING_FEW_SHOT = False


def load_data(suffix: str, need_configs: bool = False) -> Tuple[List[Dict], List]:
    with open("datasets/skillset_listen_data_1000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"] if suffix != "train" else data["train"]

    with open("datasets/skillset_listen_games_1000_v2.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"test_{suffix}"] if suffix != "train" else games["train"]
    game_layouts = ["room_door_key" in game["config"] for game in games]
    if need_configs:
        game_configs = [game["config"] for game in games]
    del games

    if need_configs:
        return data, game_configs, game_layouts
    return data, game_layouts


def get_open_file(filename: str):
    if os.path.exists(filename):
        if os.path.getsize(filename) > 0:
            return open(filename, "a")
        else:
            return open(filename, "w")
    else:
        return open(filename, "w")


def break_up_primitive_skills(is_rdk: bool) -> str:
    ns = ""
    primitive_desc = Primitive.describe()
    matches = re.findall(r"\(\d+\) (\w+: .*?);|.", primitive_desc)
    number = 1
    for match in matches:
        if len(match) > 0:
            addl = f"{number}. {match.strip()}\n"
            if number == 5 and is_rdk:  # adjust the toggle description based on RDK vs TI
                addl.replace(", opening a box, or fixing a bridge", " or opening a box")
            ns += addl
            number += 1
    ns += "7. " + re.search(r'\(7\) (.*). Note', primitive_desc).group(1) + "\n"
    ns += primitive_desc.split(".")[-2].strip() + "."
    return ns


def find_start_idx(out_file: str, data_length: int):
    try:
        start_idx = -1
        with open(out_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            match = re.search(r"Datapoint (\d+)", line)
            if match:
                start_idx = int(match.group(1))
        if start_idx == data_length - 1:
            print("Done!!!")
            return None
        start_idx += 1
    except (FileNotFoundError, IndexError):
        start_idx = 0
    print("Start idx:", start_idx)
    time.sleep(2)
    return start_idx


def build_few_shot_prompt(datapoint: Dict, task: str, is_rdk: bool, obs_window: List[str] = [], acts_window: List[str] = []) -> Union[str, List[str]]:
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    if task == "speaker":
        prompt += "You will be given a trajectory consisting of observations (obs) that you've previously seen and the corresponding actions you took in response. Your job is to identify which task the trajectory exemplifies." + "\n\n"
        prompt += "Below are three examples of trajectories you might see and the correct tasks associated with them." + "\n\n"
        training_examples = [TRAINING_DATA[i] for i in np.random.choice(NUM_TRAINING_EXAMPLES, size = 3)]
        for i, te in enumerate(training_examples):
            prompt += f"EXAMPLE {i + 1}\n"
            prompt += "TRAJECTORY:\n"
            for j, (obs, act) in enumerate(zip(te["partial_text_obs"], te["actions"])):
                prompt += f"Obs {j + 1}: {obs}\nYour action: {act}\n"
            prompt += f"Final obs: {te['partial_text_obs'][-1]}" + "\n"
            prompt += f"TASK: {te['skill_name']}\n"
            prompt += f"END EXAMPLE {i + 1}\n\n"
        prompt += f"NOW IT IS YOUR TURN. Here is your trajectory:\n"
        for i, (obs, act) in enumerate(zip(datapoint["partial_text_obs"], datapoint["actions"])):
            prompt += f"Obs {i + 1}: {obs}\nYour action: {act}\n"
        prompt += f"Final obs: {datapoint['partial_text_obs'][-1]}" + "\n\n"
        prompt += "Which of the following tasks does your above trajectory exemplify? Below are the available task names, descriptions, and examples. Your answer should be the exact task name (including underscores) taken ONLY from the below options, such as 'primitive', 'drop_at', etc. No more, no less." + "\n"
        for i, skill in enumerate(Skills):
            prompt += f"{i + 1}. {skill.name}: {skill.value.describe()}\n"
        prompt += "\nYour answer: "
    elif task == "listener":
        prompt += "You will be given a task to execute"
        if len(obs_window) == 1:
            prompt += ", as well as an initial observation (obs) of what you currently see in the environment. Given these, your job is to choose the best action to take in order to make progress towards or complete your task." + "\n\n"
            prompt += "Below are three examples of tasks you might receive, initial observations you might see, and the expected actions you should take." + "\n\n"
        else:
            if len(obs_window) == 2:
                prompt += ". You will also be given a previous observation (obs) that you've seen in the environment and the corresponding action you took in response. "
            elif len(obs_window) == 3:
                prompt += ". You will also be given previous observations (obs) that you've seen in the environment and the corresponding actions you took in response. "
            prompt += "Given these, and the current observation, your job is to choose the best action to take in order to make progress towards or complete your task." + "\n\n"
            prompt += "Below are three examples of tasks you might receive, observations and actions you might have, and the expected actions you should take." + "\n\n"
        training_examples = [TRAINING_DATA[i] for i in np.random.choice(NUM_TRAINING_EXAMPLES, size = 3)]
        for i, te in enumerate(training_examples):
            prompt += f"EXAMPLE {i + 1}\n"
            prompt += f"TASK: {te['instruction']}. This means to {to_enum(Skills, datapoint['skill_name']).value.describe()}\n"
            if len(obs_window) == 2:
                prompt += f"PREVIOUS OBS: {te['partial_text_obs'][0]}\nYOUR ACTION: {te['actions'][0]}\n"
            elif len(obs_window) == 3:
                prompt += f"OBS TWO TIMESTEPS AGO: {te['partial_text_obs'][0]}\nYOUR ACTION: {te['actions'][0]}\nOBS ONE TIMESTEP AGO: {te['partial_text_obs'][1]}\nYOUR ACTION: {te['actions'][1]}\n"
            prompt += f"CURRENT OBS: {te['partial_text_obs'][2]}\n"
            prompt += f"CORRECT ACTION: {te['actions'][2]}\n"
            prompt += f"END EXAMPLE {i + 1}\n\n"
        prompt += f"NOW IT IS YOUR TURN. Your task is: {datapoint['instruction']}. This means to {to_enum(Skills, datapoint['skill_name']).value.describe()}\n"
        if len(obs_window) == 2:
            prompt += f"PREVIOUS OBS: {obs_window[0]}\nYOUR ACTION: {acts_window[0]}\n"
        elif len(obs_window) == 3:
            prompt += f"OBS TWO TIMESTEPS AGO: {obs_window[0]}\nYOUR ACTION: {acts_window[0]}\nOBS ONE TIMESTEP AGO: {obs_window[1]}\nYOUR ACTION: {acts_window[1]}\n"
        prompt += f"CURRENT OBS: {obs_window[-1]}\n\n"
        prompt += "Which of the following actions should you take to make progress towards or complete your given task? Below are the action names and their descriptions. Your answer should be the exact action name (NOT number) chosen ONLY from the list below, such as 'left', 'forward', etc. No more, no less." + "\n"
        prompt += break_up_primitive_skills(is_rdk)
        prompt += "\nYour answer: "
    return prompt


def build_prompt(datapoint: Dict, task: str, is_rdk: bool, obs_window: List[str] = [], acts_window: List[str] = []) -> str:
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    if task == "speaker":
        prompt += """
        Below is a trajectory of observations (obs) that you've previously seen and the corresponding actions you took in response.
        """.strip() + "\n"
        for i, (obs, act) in enumerate(zip(datapoint["partial_text_obs"], datapoint["actions"])):
            prompt += f"Obs {i + 1}: {obs}\nYour action: {act}\n"
        prompt += f"Final obs: {datapoint['partial_text_obs'][-1]}" + "\n\n"
        prompt += """
        Which of the following tasks does your above trajectory execute? Below are the task names, descriptions, and examples. Your answer should be the exact task name (including underscores), such as 'primitive', 'drop_at', etc. No more, no less.
        """.strip() + "\n"
        for i, skill in enumerate(Skills):
            prompt += f"{i + 1}. {skill.name}: {skill.value.describe()}\n"
        prompt += "\nYour answer: "
    elif task == "listener":
        prompt += f"Your task is: {datapoint['instruction']}. This means to {to_enum(Skills, datapoint['skill_name']).value.describe()}" + "\n\n"
        if len(obs_window) == 2:
            prompt += """
            Below is the previous observation (obs) that you've seen and the corresponding action you took in response.
            """.strip() + "\n"
            prompt += f"Previous obs: {obs_window[0]}\nYour action: {acts_window[0]}\n\n"
        elif len(obs_window) == 3:
            prompt += """
            Below are the past two observations (obs) that you've seen and the corresponding actions you took in response.
            """.strip() + "\n"
            prompt += f"Obs two timesteps ago: {obs_window[0]}\nYour action: {acts_window[0]}\nObs one timestep ago: {obs_window[1]}\nYour action: {acts_window[1]}\n\n"
        prompt += f"Your current observation: {obs_window[-1]}" + "\n\n"
        prompt += """
        Which of the following actions should you take to make progress towards or complete your given task? Below are the action names and their descriptions. Your answer should be the exact action name (NOT number) such as 'left', 'forward', etc. No more, no less.
        """.strip() + "\n"
        prompt += break_up_primitive_skills(is_rdk)
        prompt += "\nYour answer: "
    return prompt


def speaker_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a trajectory, can the agent say what skill is being executed?
    """
    data, game_layouts = load_data(suffix)
    start_idx = find_start_idx(out_file, len(data))
    if start_idx is None:
        return
    
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                if USING_FEW_SHOT:
                    prompt = build_few_shot_prompt(datapoint, "speaker", game_layouts[game_id])
                else:
                    prompt = build_prompt(datapoint, "speaker", game_layouts[game_id])
                f.write(prompt)
                return
                resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 20)
                model_answer = json.loads(resp.json())["output"]["text"]
                correct_answer = datapoint["skill_name"]
                f.write(f"Datapoint {i}: correct: {correct_answer}, model: {model_answer}\n")
                if i % 5 == 0:
                    time.sleep(1)
        f.write("DONE")


def listener_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a skill instruction, can the agent construct a trajectory?
    """
    data, game_configs, game_layouts = load_data(suffix, need_configs = True)
    start_idx = find_start_idx(out_file, len(data))
    if start_idx is None:
        return

    counter = 0
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                obs_window = [datapoint["partial_text_obs"][0]]
                act_window = []
                max_steps = len(datapoint["actions"]) + 10
                curr_steps = 0
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                game_config = make_config(config_str = game_configs[game_id])
                env = make_env(getattr(game_config, game_config.roles.executor).world_model)
                env.reset()
                pbar = tqdm(total = max_steps)
                while curr_steps < max_steps:
                    if USING_FEW_SHOT:
                        prompt = build_few_shot_prompt(datapoint, "listener", game_layouts[game_id], obs_window, act_window)
                    else:
                        prompt = build_prompt(datapoint, "listener", game_layouts[game_id], obs_window, act_window)
                    if curr_steps == 3:
                        f.write(prompt)
                        return
                    # resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 10)
                    # model_answer = json.loads(resp.json())["output"]["text"]
                    model_answer = "left"
                    match = re.search(r"(left|right|forward|pickup|drop|toggle|done)", model_answer.lower())
                    if match:
                        if match.group(1) == "done":
                            # f.write(f"Datapoint {i} prompt {curr_steps}: model: done\n")
                            break
                        action = ACTION_TO_IDX[match.group(1)]
                        env.step(action)
                        obs_window.append(describe_state(MindGridEnvState(env)))
                        act_window.append(IDX_TO_ACTION[action])
                        if len(obs_window) == 4:
                            obs_window = obs_window[1:]
                            act_window = act_window[1:]
                        # f.write(f"Datapoint {i} prompt {curr_steps}: model: {IDX_TO_ACTION[action]}\n")
                        curr_steps += 1
                        pbar.update(1)
                    else:
                        f.write(f"Datapoint {i} prompt {curr_steps}: model: FAIL\n")
                        pbar.n = max_steps
                        pbar.refresh()
                        break
                    counter += 1
                    if counter % 5 == 0:
                        time.sleep(1)
                pbar.close()
        f.write("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", "-s", action = "store_true", default = False)
    parser.add_argument("--listener", "-l", action = "store_true", default = False)
    parser.add_argument("--ood", "-ood", action = "store_true", default = False)
    parser.add_argument("--id", "-id", action = "store_true", default = False)
    parser.add_argument("--model", "-m", type = int, required = True)
    parser.add_argument("--few_shot", "-f", action = "store_true", default = False)
    args = parser.parse_args()
    
    assert args.speaker or args.listener, "Choose either speaker or listener"
    assert args.ood or args.id, "Choose either _in or _out"
    USING_FEW_SHOT = args.few_shot
    print("Using few shot?", USING_FEW_SHOT)
    time.sleep(2)
    if USING_FEW_SHOT:
        TRAINING_DATA, _ = load_data("train")
        TRAINING_DATA = [dp for dp in TRAINING_DATA if 3 <= len(dp["actions"]) <= 10]  # need ≥ three for listener task, not too many for speaker task
        NUM_TRAINING_EXAMPLES = len(TRAINING_DATA)

    # output_file = f"intention_{'speaker' if args.speaker else 'listener'}_{'id' if args.id else 'ood'}.txt"
    # output_file = output_file.replace(".txt", f"_{MODELS[args.model].split('-')[0]}.txt")
    output_file = "intention_prompts.txt"
    if "id_llama" in output_file:
        llmengine.api_engine.api_key = SCALE_KEY
    elif "ood_llama" in output_file:
        llmengine.api_engine.api_key = BACKUP_SCALE_KEY
    elif "id_mixtral" in output_file:
        llmengine.api_engine.api_key = BACKUP_BACKUP_SCALE_KEY
    elif "ood_mixtral" in output_file:
        llmengine.api_engine.api_key = BBB_SCALE_KEY
    elif "id_gemma" in output_file:
        llmengine.api_engine.api_key = BBBB_SCALE_KEY
    elif "ood_gemma" in output_file:
        llmengine.api_engine.api_key = BBBBB_SCALE_KEY

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "few_shot" if USING_FEW_SHOT else "zero_shot", output_file)
    if args.speaker:
        speaker_task(output_path, "out" if args.ood else "in", args.model)
    elif args.listener:
        listener_task(output_path, "out" if args.ood else "in", args.model)
