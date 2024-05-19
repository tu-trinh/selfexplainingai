import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mindgrid.skills import Skills
from mindgrid.skills.skills import Primitive
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.env import MindGridEnvState
from mindgrid.infrastructure.env_utils import describe_state
from mindgrid.infrastructure.env_constants import ACTION_TO_IDX, IDX_TO_ACTION
from mindgrid.envs.edits import Edits

import argparse
import re
import numpy as np
import pandas as pd
import pickle


MODELS = ["llama", "mixtral", "gemma"]
DISTS = ["in", "out"]


def load_data(prefix, suffix):
    with open(f"datasets/{prefix}_listen_data_{5 if prefix == 'worldmodel' else 1}000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"]
    with open(f"datasets/{prefix}_listen_games_{5 if prefix == 'worldmodel' else 1}000_v2.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"test_{suffix}"]
    return data, games


def get_lines(mismatch, task, dist, model):
    filename = f"{mismatch}_{task}_{'id' if dist == 'in' else 'ood'}_{model}.txt"
    try:
        with open(os.path.join(output_dir, filename), "r") as f:
            lines = f.readlines()
            if "DONE" not in lines[-1]:
                raise ValueError
        return lines
    except (ValueError, FileNotFoundError):
        print(filename, "does not exist!")
        return


def evaluate_intention(args, output_dir):
    if args.listener:
        evaluate_intention_listener(output_dir)
    elif args.speaker:
        evaluate_intention_speaker(output_dir)


def evaluate_intention_listener(output_dir):
    """
    Given a skill instruction, can the agent construct a trajectory?
    """
    csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    for dist in DISTS:
        data, games = load_data("skillset", dist)
        for model in MODELS:
            lines = get_lines("intention", "listener", dist, model)
            score = []  # +1 if final env state matches ground truth, +0 otherwise
            curr_dp = 0
            action_seq = []
            i = 0
            while i < len(lines):
                if f"Datapoint {curr_dp}" in lines[i]:
                    curr_action = lines[i].split(":")[-1].strip()
                    action_seq.append(curr_action)
                else:
                    if len(action_seq) > 0:
                        if action_seq[-1] != "FAIL":
                            game_id = int(re.search(r"(\d+)", data[curr_dp]["game_id"]).group(1))
                            game_config = make_config(config_str = games[game_id]["config"])
                            env = make_env(getattr(game_config, game_config.roles.executor).world_model)
                            env.reset()
                            for action in data[curr_dp]["actions"]:
                                env.step(ACTION_TO_IDX[action])
                            final_true_obs = describe_state(MindGridEnvState(env))
                            env.reset()
                            for action in action_seq:
                                env.step(ACTION_TO_IDX[action])
                            final_pred_obs = describe_state(MindGridEnvState(env))
                            score.append(final_true_obs.strip() == final_pred_obs.strip())
                        else:
                            score.append(0)
                    curr_dp += 1
                    action_seq = [lines[i].split(":")[-1].strip()]
                i += 1
            
            assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
            csv.loc[(csv["mismatch"] == "intention") & (csv["task"] == "listener")
                    & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score)
            print("Finished", dist, model)
    csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)


def evaluate_intention_speaker(output_dir):
    """
    Given a trajectory, can the agent say what skill is being executed?
    """
    csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    for dist in DISTS:
        for model in MODELS:
            lines = get_lines("intention", "speaker", dist, model) + ["Datapoint"]  # sentinel
            score = []  # +1 if skill name matches, +0 otherwise
            dp_builder = lines[0].replace("\n", " ")
            for i in range(1, len(lines)):
                if "Datapoint" in lines[i]:
                    if len(dp_builder) > 0:
                        matches = re.search(r"correct: (.*), model: (.*)", dp_builder)
                        if matches:
                            score.append(matches.group(1).lower().strip() in matches.group(2).lower())
                    dp_builder = lines[i].replace("\n", " ")
                else:
                    dp_builder += lines[i].replace("\n", " ")
            # assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
            csv.loc[(csv["mismatch"] == "intention") & (csv["task"] == "speaker")
                    & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score)
            print("Finished", dist, model)
    csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)


def evaluate_belief(args, output_dir):
    if args.listener:
        evaluate_belief_listener(output_dir)
    elif args.speaker:
        evaluate_belief_speaker(output_dir)


def evaluate_belief_listener(output_dir):
    """
    Given a trajectory and environment edits, can the agent tell how the observations will change?
    """
    csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    for dist in DISTS:
        data, _ = load_data("worldmodel", dist)
        for model in MODELS:
            lines = get_lines("belief", "listener", dist, model)
            score = []  # +x where x is proportion of questions in a query correct
            matches = re.search(r"Datapoint (\d+) query (\d+)", lines[0])
            dp_idx = int(matches.group(1))
            q_idx = int(matches.group(2))
            dp_builder = lines[0]
            for i in range(1, len(lines)):
                if "Datapoint" in lines[i] and "query" in lines[i]:
                    if len(dp_builder) > 0:
                        # Process current datapoint
                        true_answers = [question["answer"] for question in data[dp_idx]["queries"][q_idx]]
                        dp_builder = re.sub(r"Datapoint (\d+) query (\d+): model:", "", dp_builder).strip()
                        pred_answers = re.split(r"[,\n\s]+", dp_builder)
                        sub_score = 0
                        for j in range(len(true_answers)):
                            sub_score += true_answers[j].lower().strip() in pred_answers[j].lower()
                        score.append(sub_score / len(true_answers))
                    # Get next datapoint
                    dp_builder = lines[i]
                    matches = re.search(r"Datapoint (\d+) query (\d+)", dp_builder)
                    dp_idx = int(matches.group(1))
                    q_idx = int(matches.group(2))
                else:
                    dp_builder += lines[i]
            assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
            csv.loc[(csv["mismatch"] == "belief") & (csv["task"] == "listener")
                    & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score)
            print("Finished", dist, model)
    csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)


def evaluate_belief_speaker(output_dir):
    """
    Given a trajectory and its own environment description, can the agent tell how the other environment has been changed?
    """
    csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    for dist in DISTS:
        data, _ = load_data("worldmodel", dist)
        for model in MODELS:
            lines = get_lines("belief", "speaker", dist, model) + ["Datapoint"]  # sentinel
            score = []  # +x where x is fraction of edits matched
            dp_builder = lines[0].replace("\n", " ")
            curr_dp = int(re.search(r"Datapoint (\d+)", lines[0]).group(1))
            for i in range(1, len(lines)):
                if "Datapoint" in lines[i]:
                    if len(dp_builder) > 0:
                        true_edits = set(data[curr_dp]["edit_descriptions"])
                        pred_edits = [edit.replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("assistant", "").replace("<|end_header_id|>", "").replace("**", "") for edit in dp_builder.split(",")]
                        pred_edits = set([pred_edit for pred_edit in pred_edits if len(pred_edit) != 0])
                        matches = true_edits.intersection(pred_edits)
                        extras = pred_edits.difference(true_edits)
                        missing = true_edits.difference(pred_edits)
                        score.append(min(0, len(matches) - len(extras) - len(missing)) / len(true_edits))
                    curr_dp = int(re.search(r"Datapoint (\d+)", lines[i]).group(1))
                    dp_builder = lines[i].replace("\n", " ")
                else:
                    dp_builder += lines[i].replace("\n", " ")
            assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
            csv.loc[(csv["mismatch"] == "belief") & (csv["task"] == "speaker")
                    & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score)
            print("Finished", dist, model)
    csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intention", "-i", action = "store_true", default = False)
    parser.add_argument("--belief", "-b", action = "store_true", default = False)
    parser.add_argument("--speaker", "-s", action = "store_true", default = False)
    parser.add_argument("--listener", "-l", action = "store_true", default = False)
    args = parser.parse_args()

    assert args.intention or args.belief, "Choose intention or belief"
    assert args.speaker or args.listener, "Choose speaker or listener"

    output_dir = os.path.dirname(os.path.abspath(__file__))
    if args.intention:
        evaluate_intention(args, output_dir)
    elif args.belief:
        evaluate_belief(args, output_dir)
