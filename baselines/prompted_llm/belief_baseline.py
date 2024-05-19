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
import re
from typing import Dict, List, Tuple, Union
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


def load_data(suffix: str, need_configs: bool = False) -> Tuple[List[Dict], List]:
    with open("datasets/worldmodel_listen_data_5000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"]

    with open("datasets/worldmodel_listen_games_5000_v2.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"test_{suffix}"]
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


def build_prompt(datapoint: Dict, task: str, is_rdk: bool, env: MindGridEnv = None, obs_window: List[str] = [], acts_window: List[str] = []) -> Union[str, List[str]]:
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
        for i, query in enumerate(datapoint["queries"][0]):
            prompt += f"{i + 1}. {query['question'].capitalize()}\n"
        prompt += "\nYour answer: "
    return prompt


def speaker_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a trajectory and its own environment description, can the agent tell how the other environment has been changed?
    """
    data, game_configs, game_layouts = load_data(suffix, need_configs = True)
    start_idx = find_start_idx(out_file, len(data))
    if not start_idx:
        return
    
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                game_config = make_config(config_str = game_configs[game_id])
                env = make_env(getattr(game_config, game_config.roles.executor).world_model)
                env.reset()
                prompt = build_prompt(datapoint, "speaker", game_layouts[game_id], env = env)
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
    start_idx = find_start_idx(out_file, len(data))
    if not start_idx:
        return

    counter = 0
    with get_open_file(out_file) as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                obs_window, acts_window = [], []
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                for j in tqdm(range(len(datapoint["queries"]))):
                    prompt = build_prompt(datapoint, "listener", game_layouts[game_id], obs_window = obs_window, acts_window = acts_window)
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
    args = parser.parse_args()
    
    assert args.speaker or args.listener, "Choose either speaker or listener"
    assert args.ood or args.id, "Choose either _in or _out"

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
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    if args.speaker:
        speaker_task(output_path, "out" if args.ood else "in", args.model)
    elif args.listener:
        listener_task(output_path, "out" if args.ood else "in", args.model)
