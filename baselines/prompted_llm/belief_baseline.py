import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mindgrid.skills import Skills
from mindgrid.skills.skills import Primitive
from package.infrastructure.access_tokens import SCALE_KEY, BACKUP_SCALE_KEY
from mindgrid.infrastructure.basic_utils import to_enum

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
You are an agent who is trying to complete a task in a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has four special rules you must keep in mind:
1. You can only unlock a locked door with a key of the same color as the door
2. You can pick up keys, balls, and boxes, but you can only be carrying one object at a time
3. You can only put an object down in a cell that has no other object
4. When you open a box, it disappears and is replaced by the object inside it, IF there is one.
"""
TI_INTRO = """
You are an agent who is trying to complete a task in a grid-like environment, where the distance between each cell in the grid and its neighbor is one step. This environment has five special rules you must keep in mind:
1. You can pick up keys, balls, boxes, hammers, and fireproof shoes, but you can only be carrying one object at a time (a pair of shoes counts as one object)
2. If you step in lava you die instantly, but if the lava is cooled or if you are carrying fireproof shoes you will be fine
3. You can safely cross bridges if they are not damaged; you can fix damaged bridges with a hammer
4. You can only put an object down in a cell that has no other object
5. When you open a box, it disappears and is replaced by the object inside it, IF there is one.
"""


def load_data(suffix: str) -> Tuple[List[Dict], List]:
    with open("datasets/worldmodel_listen_data_1000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"]

    with open("datasets/worldmodel_listen_games_1000_v2.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"test_{suffix}"]
    game_layouts = ["room_door_key" in game["config"] for game in games]
    del games

    return data, game_layouts


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


def build_prompt(datapoint: Dict, task: str, is_rdk: bool) -> Union[str, List[str]]:
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
        return prompt
    elif task == "listener":
        prompts = []
        for i in range(len(datapoint["partial_text_obs"]) - 1):
            prompt = f"Your task is: {datapoint['instruction']}. This means to {to_enum(Skills, datapoint['skill_name']).value.describe()}" + "\n\n"
            if i == 1:
                prompt += """
                Below is the previous observation (obs) that you've seen and the corresponding action you took in response.
                """.strip() + "\n"
                prompt += f"Previous obs: {datapoint['partial_text_obs'][i - 1]}\nYour action: {datapoint['actions'][i - 1]}\n\n"
            elif i >= 2:
                prompt += """
                Below are the past two observations (obs) that you've seen and the corresponding actions you took in response.
                """.strip() + "\n"
                prompt += f"Obs two timesteps ago: {datapoint['partial_text_obs'][i - 2]}\nYour action: {datapoint['actions'][i - 2]}\nObs one timestep ago: {datapoint['partial_text_obs'][i - 1]}\nYour action: {datapoint['actions'][i - 1]}\n\n"
            prompt += f"Your current observation: {datapoint['partial_text_obs'][i]}" + "\n\n"
            prompt += """
            Which of the following actions should you take to make progress towards or complete your given task? Your answer should be the exact action name, NOT number, such as 'left', 'forward', etc. No more, no less.
            """.strip() + "\n"
            prompt += break_up_primitive_skills(is_rdk)
            prompt += "\nYour answer: "
            prompts.append(prompt)
        return prompts


def speaker_task(out_file: str, suffix: str, model_idx: int):
    """
    Given a trajectory, can the agent say what skill is being executed?
    """
    data, game_layouts = load_data(suffix)
    llmengine.api_engine.api_key = SCALE_KEY
    
    try:
        start_idx = -1
        with open(out_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            match = re.search(r"Datapoint (\d+)", line)
            if match:
                start_idx = int(match.group(1))
        if start_idx == len(data) - 1:
            print("Done!!!")
            return
        start_idx += 1
    except (FileNotFoundError, IndexError):
        start_idx = 0
    print("Start idx:", start_idx)
    time.sleep(2)
    
    with open(out_file, "w") as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                prompt = build_prompt(datapoint, "speaker", game_layouts[game_id])
                resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 10)
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
    data, game_layouts = load_data(suffix)
    llmengine.api_engine.api_key = BACKUP_SCALE_KEY

    try:
        start_idx, start_sub_idx = -1, -1
        with open(out_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            match = re.search(r"Datapoint (\d+) prompt (\d+)", line)
            if match:
                start_idx = int(match.group(1))
                start_sub_idx = int(match.group(2))
        if start_idx == len(data) - 1:
            print("Done!!!")
            return
        if start_sub_idx == len(data[start_idx]["actions"]) - 1:  # finished one trajectory the whole way
            start_idx += 1
            start_sub_idx = 0
        else:  # need to finish it up
            start_sub_idx += 1
    except (FileNotFoundError, IndexError):
        start_idx = 0
        start_sub_idx = 0
    print("Start idx:", start_idx, "Sub idx:", start_sub_idx)
    time.sleep(3)

    counter = 0
    with open(out_file, "w") as f:
        for i, datapoint in enumerate(tqdm(data)):
            if i >= start_idx:
                game_id = int(re.search(r"(\d+)", datapoint["game_id"]).group(1))
                prompts = build_prompt(datapoint, "listener", game_layouts[game_id])
                for j, prompt in enumerate(tqdm(prompts)):
                    if j >= start_sub_idx:
                        resp = Completion.create(model = MODELS[model_idx], prompt = prompt, temperature = TEMPERATURE, max_new_tokens = 2)
                        model_answer = json.loads(resp.json())["output"]["text"]
                        correct_answer = datapoint["actions"][j]
                        f.write(f"Datapoint {i} prompt {j}: correct: {correct_answer}, model: {model_answer}\n")
                        counter += 1
                        if counter % 5 == 0:
                            time.sleep(1)
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

    output_file = f"intention_{'speaker' if args.speaker else 'listener'}_{'id' if args.id else 'ood'}.txt"
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file.replace(".txt", f"_{MODELS[args.model].split('-')[0]}.txt"))
    if args.speaker:
        speaker_task(output_path, "out" if args.ood else "in", args.model)
    elif args.listener:
        listener_task(output_path, "out" if args.ood else "in", args.model)
