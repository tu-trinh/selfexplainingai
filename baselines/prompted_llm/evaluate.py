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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem


MODELS = ["llama", "mixtral", "gemma"]
DISTS = ["in", "out"]
SHOTS = {"zero_shot": 0, "few_shot": 3}
LLAMA_REMOVALS = ["<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>"]


def load_data(prefix, suffix):
    with open(f"datasets/{prefix}_listen_data_{5 if prefix == 'worldmodel' else 1}000_v2.pickle", "rb") as f:
        data = pickle.load(f)
    data = data[f"test_{suffix}"]
    with open(f"datasets/{prefix}_listen_games_{5 if prefix == 'worldmodel' else 1}000_v2.pickle", "rb") as f:
        games = pickle.load(f)
    games = games[f"test_{suffix}"]
    return data, games


def get_lines(shot, mismatch, task, dist, model):
    filename = f"{mismatch}_{task}_{'id' if dist == 'in' else 'ood'}_{model}.txt"
    try:
        with open(os.path.join(output_dir, shot, filename), "r") as f:
            lines = f.readlines()
            if "DONE" not in lines[-1]:
                raise ValueError
        return lines
    except ValueError:
        print(filename, "may not be done!")
    except FileNotFoundError:
        print(filename, "does not exist!")
        return


def clean_model_output(output, model):
    if model == "llama":
        for llama_removal in LLAMA_REMOVALS:
            output = output.replace(llama_removal, "")
    elif model == "gemma":
        output = output.replace("**", "")
    return output


def evaluate_intention(args, output_dir):
    if args.listener:
        evaluate_intention_listener(output_dir)
    elif args.speaker:
        evaluate_intention_speaker(output_dir)


def evaluate_intention_listener(output_dir):
    """
    Given a skill instruction, can the agent construct a trajectory?
    """
    # csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    skill_metrics = {}
    for skill in Skills:
        skill_metrics[skill.name] = []  # 0/1 trajectory completed
    for shot in SHOTS:
        for dist in DISTS:
            data, games = load_data("skillset", dist)
            for model in MODELS:
                lines = get_lines(shot, "intention", "listener", dist, model)
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
                                if final_true_obs.strip() == final_pred_obs.strip():
                                    score.append(1)
                                    if model == "llama":
                                        skill_metrics[data[curr_dp]["skill_name"]].append(1)
                            else:
                                score.append(0)
                                if model == "llama":
                                    skill_metrics[data[curr_dp]["skill_name"]].append(0)
                        curr_dp += 1
                        action_seq = [lines[i].split(":")[-1].strip()]
                    i += 1
                # assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
                # csv.loc[(csv["mismatch"] == "intention") & (csv["task"] == "listener") & (csv["shot"] == SHOTS[shot])
                #         & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score).round(5)
                print("Finished", shot, dist, model)
    # csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)
    props = [(skill, np.mean(skill_metrics[skill])) for skill in skill_metrics]  # using mean but really this is just proportion since data is 0/1
    props.sort(key = lambda t: t[1], reverse = True)
    plot_metrics(props, "Skills", "Proportion", "Proportion of Trajectories Completed Successfully", "il_llama.png", "darkmagenta")


def evaluate_intention_speaker(output_dir):
    """
    Given a trajectory, can the agent say what skill is being executed?
    """
    # csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    skill_metrics = {}
    for skill in Skills:
        skill_metrics[skill.name] = [0, 0]  # (num identified, num appeared)
    for shot in SHOTS:
        for dist in DISTS:
            for model in MODELS:
                lines = get_lines(shot, "intention", "speaker", dist, model) + ["Datapoint"]  # sentinel
                score = []  # +1 if skill name matches, +0 otherwise
                dp_builder = lines[0].replace("\n", " ")
                for i in range(1, len(lines)):
                    if "Datapoint" in lines[i]:
                        if len(dp_builder) > 0:
                            dp_builder = clean_model_output(dp_builder, model)
                            matches = re.search(r"correct: (.*), model: (.*)", dp_builder)
                            if matches:
                                correct = matches.group(1).lower().strip()
                                model_answer = matches.group(2).lower()
                                if model == "llama":
                                    skill_metrics[correct][1] += 1
                                if correct in model_answer:
                                    score.append(1)
                                    if model == "llama":
                                        skill_metrics[correct][0] += 1
                                else:
                                    score.append(0)
                        if lines[i] != "Datapoint":
                            dp_builder = lines[i].replace("\n", " ")
                        else:  # reached sentinel
                            break
                    else:
                        dp_builder += lines[i].replace("\n", " ")
                # assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
                # csv.loc[(csv["mismatch"] == "intention") & (csv["task"] == "speaker") & (csv["shot"] == SHOTS[shot])
                #         & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score).round(5)
                print("Finished", shot, dist, model)
    # csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)
    props = [(skill, skill_metrics[skill][0] / skill_metrics[skill][1]) for skill in skill_metrics]
    props.sort(key = lambda t: t[1], reverse = True)
    plot_metrics(props, "Skills", "Proportion", "Proportion of Times Identified Correctly", "is_llama.png", "darkmagenta")


def evaluate_belief(args, output_dir):
    if args.listener:
        evaluate_belief_listener(output_dir)
    elif args.speaker:
        evaluate_belief_speaker(output_dir)


def evaluate_belief_listener(output_dir):
    """
    Given a trajectory and environment edits, can the agent tell how the observations will change?
    """
    # csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    edit_metrics = {}
    for edit in Edits:
        edit_metrics[edit.name] = []  # (sub)score when the edit appears
    for shot in SHOTS:
        for dist in DISTS:
            data, _ = load_data("worldmodel", dist)
            for model in MODELS:
                lines = get_lines(shot, "belief", "listener", dist, model) + ["Datapoint query"]  # sentinel
                score = []  # +x where x is proportion of questions in a query correct
                already_seen  = set()
                matches = re.search(r"Datapoint (\d+) query (\d+)", lines[0])
                dp_idx = int(matches.group(1))
                q_idx = int(matches.group(2))
                dp_builder = lines[0]
                for i in range(1, len(lines)):
                    if "Datapoint" in lines[i] and "query" in lines[i]:
                        if len(dp_builder) > 0 and (dp_idx, q_idx) not in already_seen:
                            # Process current datapoint
                            true_answers = [question["answer"] for question in data[dp_idx]["queries"][q_idx]]
                            dp_builder = re.sub(r"Datapoint (\d+) query (\d+): model:", "", dp_builder).strip()
                            dp_builder = clean_model_output(dp_builder, model)
                            dp_builder = dp_builder.replace("1.", " ").replace("2.", " ").replace("3.", " ").replace("4.", " ").replace("5.", " ")
                            # Try splitting on newlines...
                            pred_answers = dp_builder.split("\n")
                            pred_answers = [pred_answer.strip() for pred_answer in pred_answers if len(pred_answer.strip()) > 0]
                            newline_score = 0
                            for j in range(min(len(true_answers), len(pred_answers))):
                                newline_score += str(true_answers[j]).lower().strip() in str(pred_answers[j]).lower()
                            # ...and commas...
                            pred_answers = dp_builder.split(",")
                            pred_answers = [pred_answer.strip() for pred_answer in pred_answers if len(pred_answer.strip()) > 0]
                            comma_score = 0
                            for j in range(min(len(true_answers), len(pred_answers))):
                                comma_score += str(true_answers[j]).lower().strip() in str(pred_answers[j]).lower()
                            # ...and see which one gets higher score (probably the right one to split on)
                            sub_score = max(newline_score, comma_score)
                            score.append(sub_score / len(true_answers))
                            if model == "llama":
                                for edit in data[dp_idx]["edits"]:
                                    edit_metrics[edit].append(sub_score / len(true_answers))
                            already_seen.add((dp_idx, q_idx))
                        # Get next datapoint
                        try:
                            dp_builder = lines[i]
                            matches = re.search(r"Datapoint (\d+) query (\d+)", dp_builder)
                            dp_idx = int(matches.group(1))
                            q_idx = int(matches.group(2))
                        except:  # reached sentinel
                            break
                    else:
                        dp_builder += lines[i]
                # assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
                # csv.loc[(csv["mismatch"] == "belief") & (csv["task"] == "listener") & (csv["shot"] == SHOTS[shot])
                #         & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score).round(5)
                print("Finished", shot, dist, model)
    # csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)
    scores = [(edit, np.mean(edit_metrics[edit])) for edit in edit_metrics]
    stds = [sem(edit_metrics[edit]) for edit in edit_metrics]
    scores.sort(key = lambda t: t[1], reverse = True)
    plot_metrics(scores, "Edits", "Proportion", "Mean Proportion of Queries Answered Correctly", "bl_llama.png", "darkblue", stds = stds, std_color = "cornflowerblue")


def evaluate_belief_speaker(output_dir):
    """
    Given a trajectory and its own environment description, can the agent tell how the other environment has been changed?
    """
    # csv = pd.read_csv(os.path.join(output_dir, "scores.csv"))
    edit_metrics = {}
    for edit in Edits:
        edit_metrics[edit.name] = [0, 0]  # (num identified, num appeared)
    for shot in SHOTS:
        for dist in DISTS:
            data, _ = load_data("worldmodel", dist)
            for model in MODELS:
                lines = get_lines(shot, "belief", "speaker", dist, model) + ["Datapoint"]  # sentinel
                score = []  # +x where x is fraction of edits matched
                dp_builder = lines[0].replace("\n", ",")
                curr_dp = int(re.search(r"Datapoint (\d+)", lines[0]).group(1))
                already_seen = set()
                for i in range(1, len(lines)):
                    if "Datapoint" in lines[i]:
                        if len(dp_builder) > 0 and curr_dp not in already_seen:
                            true_edits = [edit.lower().strip() for edit in data[curr_dp]["edits"]]
                            dp_builder = clean_model_output(dp_builder, model)
                            pred_edits = [edit.lower() for edit in dp_builder.split(",")]
                            pred_edits = [edit for edit in pred_edits if len(edit) != 0]
                            matches = 0
                            if model == "llama":
                                for true_edit in true_edits:
                                    edit_metrics[true_edit][1] += 1
                                    if any([true_edit in pred_edit for pred_edit in pred_edits]):
                                        matches += 1
                                        edit_metrics[true_edit][0] += 1
                            score.append(matches / len(true_edits))
                            already_seen.add(curr_dp)
                        try:
                            curr_dp = int(re.search(r"Datapoint (\d+)", lines[i]).group(1))
                        except:  # reached sentinel
                            break
                        dp_builder = lines[i].replace("\n", ",")
                    else:
                        dp_builder += lines[i].replace("\n", ",")
                assert len(score) == len(data), f"Score has length {len(score)} and data has length {len(data)}"
                # csv.loc[(csv["mismatch"] == "belief") & (csv["task"] == "speaker") & (csv["shot"] == SHOTS[shot])
                        # & (csv["distribution"] == dist) & (csv["model"] == model), "score"] = np.mean(score).round(5)
                print("Finished", shot, dist, model)
    # csv.to_csv(os.path.join(output_dir, "scores.csv"), index = False)
    props = [(edit, edit_metrics[edit][0] / edit_metrics[edit][1]) for edit in edit_metrics]
    props.sort(key = lambda t: t[1], reverse = True)
    plot_metrics(props, "Edits", "Proportion", "Proportion of Times Identified Correctly", "bs_llama.png", "darkblue")


def plot_metrics(metrics, item, metric, title, name, color, figsize = (6, 6), stds = None, std_color = None):
    plt.rcParams["font.family"] = "DejaVu Sans"
    _, ax = plt.subplots(figsize = figsize)
    if stds is not None:
        sns.barplot(x = [metric[1] for metric in metrics], y = [metric[0] for metric in metrics], ax = ax, color = color, xerr = stds, err_kws = {"color": std_color})
    else:
        sns.barplot(x = [metric[1] for metric in metrics], y = [metric[0] for metric in metrics], ax = ax, color = color)
    ax.set_xlabel(metric, fontsize = 12)
    ax.set_ylabel(item, fontsize = 12)
    ax.set_title(title, fontsize = 14)
    # for i, tick in enumerate(ax.get_xticklabels()):
        # tick.set_rotation(20)
        # tick.set_fontsize(8)
        # if i % 2 == 0:
        #     tick.set_y(tick.get_position()[1] + 0.02)
        # else:
        #     tick.set_y(tick.get_position()[1] - 0.02)
        # tick.set_fontname(font)
    # for tick in ax.get_yticklabels():
        # tick.set_fontname(font)
    plt.savefig(os.path.join(output_dir, name), bbox_inches = "tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intention", "-i", action = "store_true", default = False)
    parser.add_argument("--belief", "-b", action = "store_true", default = False)
    parser.add_argument("--speaker", "-s", action = "store_true", default = False)
    parser.add_argument("--listener", "-l", action = "store_true", default = False)
    parser.add_argument("--overview", "-o", action = "store_true", default = False)
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    if args.overview:
        scores = pd.read_csv(os.path.join(output_dir, "scores.csv"))
        zero = scores[scores["shot"] == 0]
        few = scores[scores["shot"] == 3]
        print("Zero shot")
        print(zero.groupby(["mismatch", "task", "model"])["score"].agg(["mean", "std"]).round(4))
        print("Few shot")
        print(few.groupby(["mismatch", "task", "model"])["score"].agg(["mean", "std"]).round(4))
    else:
        assert args.intention or args.belief, "Choose intention or belief"
        assert args.speaker or args.listener, "Choose speaker or listener"
        if args.intention:
            evaluate_intention(args, output_dir)
        elif args.belief:
            evaluate_belief(args, output_dir)
