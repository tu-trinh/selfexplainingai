import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mindgrid.envs.edits import Edits
from package.infrastructure.access_tokens import *
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.env import MindGridEnv

import llmengine
from llmengine import Completion
import argparse
import pickle
import json
import numpy as np
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import time


MODELS = ["llama-3-70b-instruct", "mixtral-8x7b-instruct", "gemma-7b-instruct"]
TEMPERATURE = 0.01
RDK_INTRO = """
You are an agent inside a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has four special rules you must keep in mind:
1. You can only unlock a locked door with a key of the same color as the door
2. You can pick up keys, balls, and boxes, but you can only be carrying one object at a time
3. You can only put an object down in a cell that has no other object
4. When you open a box, it disappears and is replaced by the object inside it, IF there is one
"""
TI_INTRO = """
You are an agent inside a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has five special rules you must keep in mind:
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
    with open("datasets/worldmodel_listen_data_5000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"] if suffix != "train" else data["train"]

    with open("datasets/worldmodel_listen_games_5000_v2.pickle", "rb") as f:
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


def find_start_idx(out_file: str, data_length: int, data: List = None):
    try:
        start_idx = -1
        with open(out_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if data is None:
                match = re.search(r"Datapoint (\d+)", line)
            else:
                match = re.search(r"Datapoint (\d+) query (\d+)", line)
            if match:
                start_idx = int(match.group(1))
                if data is not None:
                    sub_start_idx = int(match.group(2))
        if start_idx == data_length - 1:
            if data is None:
                print("Done!!!")
                return None
            else:
                queries_length = len(data[start_idx]["queries"])
                if sub_start_idx == queries_length - 1:
                    print("Done!!!")
                    return None, None
                else:
                    sub_start_idx += 1
        if data is None:
            start_idx += 1
        else:
            if start_idx != -1:
                queries_length = len(data[start_idx]["queries"])
                if sub_start_idx == queries_length - 1:
                    start_idx += 1
                    sub_start_idx = 0
                else:
                    sub_start_idx += 1
            else:
                print("Empty file")
                start_idx = 0
                sub_start_idx = 0
    except (FileNotFoundError, IndexError):
        start_idx = 0
        if data is not None:
            sub_start_idx = 0
    print("Start idx:", start_idx)
    if data is not None:
        print("Sub start idx:", sub_start_idx)
    time.sleep(3)
    if data is None:
        return start_idx
    return start_idx, sub_start_idx


def build_few_shot_prompt(datapoint: Dict, task: str, is_rdk: bool, env: MindGridEnv = None, obs_window: List[str] = [], acts_window: List[str] = [], query_idx = 0) -> str:
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    if task == "speaker":
        prompt += f"You will be given a starting description of you and your environment. Elsewhere, there is another agent in another grid-like environment that also follows the {'four' if is_rdk else 'five'} special rules above. Its environment started out just like yours but then some changes were applied, thus making it, and possibly the agent's configuration inside it, different from yours. Given your starting description and a trajectory consisting of (1) observations (obs) that the OTHER agent has seen in its own environment and (2) the corresponding actions it took in response, your job is to identify which changes were applied to your environment to result in the other agent's environment." + "\n\n"
        prompt += "Below are three examples of starting descriptions you might have, trajectories the other agent might have, and answers you'd be expected to give." + "\n\n"
        training_examples = [TRAINING_DATA[i] for i in np.random.choice(NUM_TRAINING_EXAMPLES, size = 3)]
        for i, te in enumerate(training_examples):
            prompt += f"EXAMPLE {i + 1}\n"
            prompt += f"STARTING DESCRIPTION: {te['init_description']}\n"
            prompt += f"TRAJECTORY (from POV of other agent):\n"
            for j, (obs, act) in enumerate(zip(te["partial_text_obs"], te["actions"])):
                prompt += f"Obs {j + 1}: {obs}\nAction: {act}\n"
            prompt += f"Final obs: {te['partial_text_obs'][-1]}" + "\n"
            prompt += f"ANSWER: {', '.join(te['edits'])}" + "\n"
            prompt += f"END EXAMPLE {i + 1}\n\n"
        prompt += f"NOW IT IS YOUR TURN. Here is the starting description of you and your environment: {datapoint['init_description']}" + "\n"
        prompt += f"Here is the other agent's trajectory from its POV:" + "\n"
        for i, (obs, act) in enumerate(zip(datapoint["partial_text_obs"], datapoint["actions"])):
            prompt += f"Obs {i + 1}: {obs}\nAction: {act}\n"
        prompt += f"Final obs: {datapoint['partial_text_obs'][-1]}" + "\n\n"
        prompt += "Which of the following changes were applied to your environment to result in the other agent's environment? Below are the possible changes and their descriptions. Your answer should just be a comma-separated list of the exact change or changes (including underscores) taken ONLY from the below options, such as 'flip_vertical', 'change_agent_view_size', etc. No more, no less." + "\n"
        for i, edit in enumerate(Edits):
            prompt += f"{i + 1}. {edit.name}: {edit.value.describe(env)}\n"
        prompt += "\nYour answer: "
    elif task == "listener":
        prompt += f"You will be given a starting description of you and your environment as well as some changes that are sequentially applied to the environment. "
        if len(obs_window) == 0:
            prompt += "Given these, your job is to answer correctly some questions about your environment." + "\n\n"
            prompt += "Below are three examples of starting descriptions you might have, changes your environment might face, questions you might get, and answers you'd be expected to give." + "\n\n"
        else:
            if len(obs_window) == 1:
                prompt += "You will also be given a previous observation (obs) you've seen in your changed environment and the corresponding action you took in response. "
            elif len(obs_window) == 2:
                prompt += "You will also be given previous observations (obs) you've seen in your changed environment and the corresponding actions you took in response. "
            prompt += "Given all these, your job is to answer correctly some questions about your environment." + "\n\n"
            prompt += "Below are three examples of starting descriptions you might have, changes your environment might face, observations and actions you might have, questions you might get, and answers you'd be expected to give." + "\n\n"
        training_examples = [TRAINING_DATA[i] for i in np.random.choice(NUM_TRAINING_EXAMPLES, size = 3)]
        for i, te in enumerate(training_examples):
            prompt += f"EXAMPLE {i + 1}\n"
            prompt += f"STARTING DESCRIPTION: {te['init_description']}\n"
            prompt += f"CHANGES:\n"
            for j, edit_desc in enumerate(te["edit_descriptions"]):
                prompt += f"{j + 1}. {edit_desc.replace('he agent', 'he agent (you)')}\n"
            if len(obs_window) == 1:
                prompt += f"PREVIOUS OBS: {te['partial_text_obs'][0]}\n"
                prompt += f"YOUR ACTION: {te['actions'][0]}\n"
            elif len(obs_window) == 2:
                prompt += f"OBS TWO TIMESTEPS AGO: {te['partial_text_obs'][0]}\n"
                prompt += f"YOUR ACTION: {te['actions'][0]}\n"
                prompt += f"OBS ONE TIMESTEP AGO: {te['partial_text_obs'][1]}\n"
                prompt += f"YOUR ACTION: {te['actions'][1]}\n"
            prompt += f"QUESTIONS:\n"
            for j, query in enumerate(te["queries"][0]):
                prompt += f"{j + 1}. {query['question'].capitalize()}\n"
            prompt += f"ANSWERS:\n"
            for j, query in enumerate(te["queries"][0]):
                prompt += f"{j + 1}. {query['answer']}\n"
            prompt += f"END EXAMPLE {i + 1}\n\n"
        prompt += f"NOW IT IS YOUR TURN. Here is the starting description of you and your environment: {datapoint['init_description']}" + "\n"
        prompt += "Now, the following changes are sequentially applied:\n"
        for i, edit_desc in enumerate(datapoint["edit_descriptions"]):
            prompt += f"{i + 1}. {edit_desc.replace('he agent', 'he agent (you)')}\n"
        if len(obs_window) == 0:
            prompt += "Please now answer the following questions about your environment. Your response should just be a numbered list of one ONE-WORD ANSWER per question. No more, no less. Do not repeat the questions, just give your answers." + "\n"
        else:
            if len(obs_window) == 1:
                prompt += "Below is a previous observation (obs) you've seen in your changed environment and the action you took.\n"
                prompt += f"Previous obs: {obs_window[0]}\nYour action: {acts_window[0]}\n"
            elif len(obs_window) == 2:
                prompt += "Below are two previous observations (obs) you've seen in your changed environment and the actions you took.\n"
                prompt += f"Obs two timesteps ago: {obs_window[0]}\nYour action: {acts_window[0]}\nObs one timestep ago: {obs_window[1]}\nYour action: {acts_window[1]}\n"
            prompt += "Given all these, please now answer the following questions about your environment. Your response should just be a numbered list of one ONE-WORD ANSWER per question. No more, no less. Do not repeat the questions, just give your answers." + "\n"
        for i, query in enumerate(datapoint["queries"][query_idx]):
            prompt += f"{i + 1}. {query['question'].capitalize()}\n"
        prompt += "\nYour answers:\n"
    return prompt


def build_prompt(datapoint: Dict, task: str, is_rdk: bool, env: MindGridEnv = None, obs_window: List[str] = [], acts_window: List[str] = [], query_idx = 0) -> str:
    prompt = (RDK_INTRO.strip() if is_rdk else TI_INTRO.strip()) + "\n\n"
    if task == "speaker":
        prompt += f"Here is a description of you and your environment: {datapoint['init_description']}" + "\n\n"
        prompt += f"""
        Elsewhere, there is another agent in another grid-like environment that also follows the {'four' if is_rdk else 'five'} special rules above. This environment started out just like yours, but then some changes were applied, thus making it, and possibly the agent's configuration inside it, different from yours. Below is a trajectory of observations (obs) that the OTHER agent has seen in its own environment and the corresponding actions it took in response.
        """.strip() + "\n"
        for i, (obs, act) in enumerate(zip(datapoint["partial_text_obs"], datapoint["actions"])):
            prompt += f"Obs {i + 1}: {obs}\nIts action: {act}\n"
        prompt += f"Final obs: {datapoint['partial_text_obs'][-1]}" + "\n\n"
        prompt += """
        Based on the above trajectory, which of the following changes were applied to your environment to result in the other agent's environment? Below are the possible changes and their descriptions. Your answer should just be a comma-separated list of the exact change or changes (including underscores), such as 'flip_vertical', 'change_agent_view_size', etc. No more, no less.
        """.strip() + "\n"
        for i, edit in enumerate(Edits):
            prompt += f"{i + 1}. {edit.name}: {edit.value.describe(env)}\n"
        prompt += "\nYour answer: "
    elif task == "listener":
        prompt += f"Here is an initial description of you and your environment: {datapoint['init_description']}" + "\n\n"
        prompt += """
        Now, the following changes are sequentially applied to your environment:
        """.strip() + "\n"
        for i, edit_desc in enumerate(datapoint["edit_descriptions"]):
            prompt += f"{i + 1}. {edit_desc.replace('he agent', 'he agent (you)')}\n"
        prompt += "\n"
        if len(obs_window) == 0:
            prompt += """
            Given the above changes, please now answer the following questions about your environment. Your response should just be a numbered list of one ONE-WORD ANSWER per question. No more, no less. Do not repeat the questions, just give your answers.
            """.strip() + "\n"
        else:
            if len(obs_window) == 1:
                prompt += """
                Below is a previous observation (obs) you've seen in your changed environment and the corresponding action you took in response.
                """.strip() + "\n"
                prompt += f"Previous obs: {obs_window[0]}\nYour action: {acts_window[0]}\n\n"
            elif len(obs_window) == 2:
                prompt += """
                Below are two previous observations (obs) you've seen in your changed environment and the corresponding actions you took in response.
                """.strip() + "\n"
                prompt += f"Obs two timesteps ago: {obs_window[0]}\nYour action: {acts_window[0]}\nObs one timestep ago: {obs_window[1]}\nYour action: {acts_window[1]}\n\n"
            prompt += """
            Given the initial environment, the changes applied to it, and your actions thus far, please now answer the following questions. Your response should just be a numbered list of one ONE-WORD ANSWER per question. No more, no less. Do not repeat the questions, just give your answers.
            """.strip() + "\n"
        for i, query in enumerate(datapoint["queries"][query_idx]):
            prompt += f"{i + 1}. {query['question'].capitalize()}\n"
        prompt += "\nYour answer: "
    return prompt


def speaker_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a trajectory and its own environment description, can the agent tell how the other environment has been changed?
    """
    data, game_configs, game_layouts = load_data(suffix, need_configs = True)
    start_idx = find_start_idx(out_file, len(data))
    if start_idx is None:
        return
    
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                game_config = make_config(config_str = game_configs[game_id])
                env = make_env(getattr(game_config, game_config.roles.executor).world_model)
                env.reset()
                prompt = build_few_shot_prompt(datapoint, "speaker", game_layouts[game_id], env = env)
                resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 50)
                model_answer = json.loads(resp.json())["output"]["text"]
                f.write(f"Datapoint {i}: model: {model_answer}\n")
                if i % 5 == 0:
                    time.sleep(1)
        f.write("DONE")


def listener_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a trajectory and environment edits, can the agent tell how the observations will change?
    """
    data, game_layouts = load_data(suffix)
    start_idx, sub_start_idx = find_start_idx(out_file, len(data), data = data)
    if start_idx is None:
        return

    counter = 0
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                obs_window, acts_window = [], []
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                for j in tqdm(range(len(datapoint["queries"]))):
                    if j >= sub_start_idx:
                        if USING_FEW_SHOT:
                            prompt = build_few_shot_prompt(datapoint, "listener", game_layouts[game_id], obs_window = obs_window, acts_window = acts_window, query_idx = j)
                        else:
                            prompt = build_prompt(datapoint, "listener", game_layouts[game_id], obs_window = obs_window, acts_window = acts_window, query_idx = j)
                        resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 30)
                        model_answer = json.loads(resp.json())["output"]["text"]
                        f.write(f"Datapoint {i} query {j}: model: {model_answer}\n\n")
                        try:
                            obs_window.append(datapoint["partial_text_obs"][j])
                            acts_window.append(datapoint["actions"][j])
                            if len(obs_window) == 3:
                                obs_window = obs_window[1:]
                                acts_window = acts_window[1:]
                            counter += 1
                            if counter % 5 == 0:
                                time.sleep(1)
                        except IndexError:  # N obs, N queries, but N - 1 actions
                            break
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
        TRAINING_DATA = [dp for dp in TRAINING_DATA if 2 <= len(dp["actions"]) <= 5]  # need ≥ two for listener task, not too many for speaker task
        NUM_TRAINING_EXAMPLES = len(TRAINING_DATA)

    output_file = f"belief_{'speaker' if args.speaker else 'listener'}_{'id' if args.id else 'ood'}.txt"
    output_file = output_file.replace(".txt", f"_{MODELS[args.model].split('-')[0]}.txt")
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
