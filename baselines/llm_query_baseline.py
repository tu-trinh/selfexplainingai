import sys
sys.path.append("/nas/ucb/tutrinh/selfexplainingai")
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.basic_utils import debug, xor, format_seconds, manhattan_distance
from package.infrastructure.env_utils import get_obs_desc
from package.infrastructure.env_constants import DIR_TO_VEC, COLOR_NAMES
from package.infrastructure.obj_constants import NAME_OBJ_MAPPING
from package.infrastructure.llm_constants import GET_SKILL_NAME_QUESTION, GET_SKILL_NAME_QUESTION_FEW_SHOT, GET_NEXT_ACTION_QUESTION, GET_NEXT_ACTION_QUESTION_FEW_SHOT, TEMPERATURE
from package.infrastructure.access_tokens import OPENAI_KEY
from package.builder import make_agents

from minigrid.wrappers import FullyObsWrapper

import numpy as np
import pandas as pd
import re
import pickle
from tqdm import tqdm
from typing import Tuple
from argparse import ArgumentParser
import openai
openai.api_key = OPENAI_KEY
import time


continuous_logging = True


def get_datasets(mismatch: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(f"datasets/{mismatch}_datasets.pkl", "rb") as f:
        full_dataset = pickle.load(f)
    training_data = pd.DataFrame(full_dataset["train"])
    validation_data = pd.DataFrame(full_dataset["val"])
    test_data = pd.DataFrame(full_dataset["test"])
    return training_data, validation_data, test_data


def process_dataset(df: pd.DataFrame, mismatch: str, task: str):
    """
    Returns the necessary dataframe for the training with just prompt and response columns
    """
    assert mismatch in ["intention", "belief"], f"Bad mismatch {mismatch}"
    assert task in ["speaker", "listener"], f"Bad task {task}"

    out_df = pd.DataFrame()

    if mismatch == "intention" and task == "speaker":
        out_df["prompt"] = df["traj_fully_obs_text"].apply(lambda x: GET_SKILL_NAME_QUESTION.format(obs_act_seq = x))
        out_df["response"] = df["skill"]
    elif mismatch == "intention" and task == "listener":
        out_df = pd.DataFrame({"prompt": [], "response": []})
        for idx, row in df.iterrows():
            skill_name = row["skill"]
            obs_act_seq = row["traj_fully_obs_text"]
            matches = re.findall(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", obs_act_seq)
            for match in matches[:1]:  # NOTE: use if model's own resultant states affect its planning
            # for match in matches:  # NOTE: use if we just want the model to map state to action directly
                add_row = {
                    "prompt": [GET_NEXT_ACTION_QUESTION.format(skill_name = skill_name, obs_desc = match[0])],
                    "response": [match[1].strip()]
                } 
                out_df = pd.concat([out_df, pd.DataFrame(add_row)], ignore_index = True)
    out_df.reset_index(inplace = True)
    out_df = out_df[["prompt", "response"]]
    return out_df


def query_model(mismatch, task, model, training_data, test_data, shot):
    if mismatch == "intention":
        if task == "speaker":
            query_model_intention_speaker(mismatch, task, model, training_data, test_data, shot)
        else:
            query_model_intention_listener(mismatch, task, model, training_data, test_data, shot)


def query_model_intention_listener(mismatch, task, model, training_data, test_data, shot):
    assert shot in ["zero", "few"]
    if "3.5" in model:
        tpm_limit = 1000000
    else:
        tpm_limit = 8000000
    
    debug("Starting to generate outputs")
    grand_outputs = ""
    current_token_count = 0
    max_steps = 24
    num_training_samples = len(training_data)
    df_sample = range(0, 500, 25)  # just take 20 samples
    if continuous_logging:
        print(f"Indices of sampled datapoints: {list(df_sample)}\n\n")
    else:
        grand_outputs += f"Indices of sampled datapoints: {list(df_sample)}\n\n"

    for idx in tqdm(df_sample):
        row = test_data.iloc[idx]
        _, a = make_agents(config_str = row["config"])
        env = FullyObsWrapper(a.world_model)
        regex_match = re.match(r"Obs \d+: ([\w ():,'\n]*?)\nAct \d+: (\w+)", row["traj_fully_obs_text"])
        curr_state = regex_match.group(1)
        skill = row["skill"]
        goal_check = get_goal_check(skill, env)

        if shot == "few":
            sample_indices = np.random.choice(num_training_samples, size = 3, replace = False)
            traj_examples = []
            skill_examples = []
            for i in sample_indices:
                traj_examples.append(training_data.iloc[i]["traj_fully_obs_text"])
                skill_examples.append(training_data.iloc[i]["skill"])
            example_prefix = "Below are three examples of tasks and corresponding sequences of a) an observation (obs) you might encounter and b) the action (act) you would correctly take in response, in order to complete the task."
            joined_examples = "\n-----\n".join([f"EXAMPLE {i + 1}:\nTASK: {skill_examples[i]}\nCORRECT SEQUENCE:\n{traj_examples[i]}" for i in range(3)])
            few_shot_examples = f"{example_prefix}\n{joined_examples}"
            full_prompt = GET_NEXT_ACTION_QUESTION_FEW_SHOT.format(few_shot_examples = few_shot_examples, skill_name = skill, obs_desc = curr_state)
        else:
            full_prompt = GET_NEXT_ACTION_QUESTION.format(skill_name = skill, obs_desc = curr_state)
        
        if continuous_logging:
            print(f"PROMPT {idx}: {skill}\n")
        else:
            grand_outputs += f"PROMPT {idx}: {skill}\n"
        obs_idx = 0
        done = False
        while not done and obs_idx < max_steps:
            tokens_needed = len(full_prompt) // 4
            current_token_count += tokens_needed
            if current_token_count > tpm_limit:
                time.sleep(60)
                current_token_count = tokens_needed
            
            messages = [{"role": "user", "content": full_prompt}]
            try:
                response_obj = openai.ChatCompletion.create(
                    model = model,
                    messages = messages,
                    temperature = TEMPERATURE
                )
                response = int(response_obj["choices"][0]["message"]["content"])
                
                if continuous_logging:
                    print(f"OBS {obs_idx} IS: {curr_state}\n")
                    if obs_idx == 0:
                        print(f"FIRST ACTION SHOULD BE: {row['actions'][0]}\n")
                    print(f"MODEL RESPONSE FOR ACTION {obs_idx} IS: {response}\n\n")
                else:
                    grand_outputs += f"OBS {obs_idx} IS: {curr_state}\n"
                    if obs_idx == 0:  # only record answer for the very first obs, lest model veers off right away
                        grand_outputs += f"FIRST ACTION SHOULD BE: {row['actions'][0]}\n"
                    grand_outputs += f"MODEL RESPONSE FOR ACTION {obs_idx} IS: {response}\n\n"

                obs, _, _, _, _ = env.step(response)
                curr_state = get_obs_desc(obs, detail = 4, carrying = env.carrying)
                full_prompt = GET_NEXT_ACTION_QUESTION.format(skill_name = skill, obs_desc = curr_state)
                done = goal_check(env, response)
                obs_idx += 1
            except Exception as e:
                print("Could not complete LLM request OR response parsing due to", e)
                break
        if done:
            if continuous_logging:
                print("SUCCESS!!!\n")
            else:
                grand_outputs += "SUCCESS!!!\n"
        if continuous_logging:
            print("\n\n\n")
        else:
            grand_outputs += "\n\n\n"
    
    if not continuous_logging:
        with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if shot == 'few' else ''}.txt", "w") as f:
            f.write(grand_outputs)
    debug("Finished generating and decoding")


def get_skill_phrase_key(skill: str) -> str:
    if skill in ["left", "turn left", "make a left turn"]:
        return "left"
    if skill in ["right", "turn right", "make a right turn"]:
        return "right"
    if skill in ["forward", "step forward", "go forward"]:
        return "forward"
    if skill in ["backward", "turn backward", "make a U-turn"]:
        return "backward"
    if skill in ["pickup", "grab", "snatch up"]:
        return "pickup"
    if skill in ["drop", "lay down", "set down"]:
        return "drop"
    if skill in ["toggle", "switch", "activate"]:
        return "toggle"
    if any(char.isdigit() for char in skill):
        if "right" in skill:
            return "move_right"
        if "left" in skill:
            return "move_left"
        if "back" in skill or "retreat" in skill:
            return "move_backward"
        return "move_forward"
    if any(term in skill for term in ["go", "proceed", "head", "approach", "towards", "navigate"]):
        return "go_"
    if any(term in skill for term in ["pickup", "acquire", "retrieve", "obtain", "fetch", "procure"]):
        return "pickup_"
    if any(term in skill for term in ["put", "down"]):
        return "put_"
    if "open" in skill:
        return "open_"
    if any(term in skill for term in ["close", "shut"]):
        return "close_"
    if any(term in skill for term in ["unl", "unb"]):
        return "unlock_"


def get_goal_check(skill, env):
    phrase_key = get_skill_phrase_key(skill)
    init_pos = env.agent_pos
    init_dir = env.agent_dir
    color = None
    target = None
    for c in COLOR_NAMES:
        if c.lower() in skill:
            color = c.lower()
            break
    for o in NAME_OBJ_MAPPING:
        if o.lower() in skill:
            target = NAME_OBJ_MAPPING[o]
            break
    if color and target:
        for obj, pos in env.objs + env.doors + env.keys:
            if type(obj) == target and obj.color == color:
                target_obj = obj
                target_pos = pos
                break

    if phrase_key == "left":
        def goal_check(env, act):
            return env.agent_dir == 3 if init_dir == 0 else env.agent_dir == init_dir - 1
    elif phrase_key == "right":
        def goal_check(env, act):
            return env.agent_dir == 0 if init_dir == 3 else env.agent_dir == init_dir + 1
    elif phrase_key == "forward":
        def goal_check(env, act):
            dir_vec = DIR_TO_VEC[init_dir]
            return init_pos[0] + dir_vec[0] == env.agent_pos[0] and init_pos[1] + dir_vec[1] == env.agent_pos[1]
    elif phrase_key == "backward":
        def goal_check(env, act):
            return env.agent_dir == init_dir + 2 if init_dir < 2 else env.agent_dir == init_dir - 2
    elif phrase_key == "pickup":
        def goal_check(env, act):
            return act == 3
    elif phrase_key == "drop":
        def goal_check(env, act):
            return act == 4
    elif phrase_key == "toggle":
        def goal_check(env, act):
            return act == 5
    elif phrase_key == "move_forward":
        def goal_check(env, act):
            num_steps = int(skill.split('_')[2])
            dir_vec = DIR_TO_VEC[init_dir]
            return init_pos[0] + dir_vec[0]*num_steps == env.agent_pos[0] and init_pos[1] + dir_vec[1]*num_steps == env.agent_pos[1]
    elif phrase_key == "move_right":
        def goal_check(env, act):
            num_steps = int(skill.split('_')[2])
            dir_vec = DIR_TO_VEC[init_dir + 1] if init_dir < 3 else DIR_TO_VEC[0]
            return init_pos[0] + dir_vec[0]*num_steps == env.agent_pos[0] and init_pos[1] + dir_vec[1]*num_steps == env.agent_pos[1]
    elif phrase_key == "move_left":
        def goal_check(env, act):
            num_steps = int(skill.split('_')[2])
            dir_vec = DIR_TO_VEC[init_dir - 1] if init_dir > 0 else DIR_TO_VEC[3]
            return init_pos[0] + dir_vec[0]*num_steps == env.agent_pos[0] and init_pos[1] + dir_vec[1]*num_steps == env.agent_pos[1]
    elif phrase_key == "move_backward":
        def goal_check(env, act):
            num_steps = int(skill.split('_')[2])
            dir_vec = DIR_TO_VEC[init_dir + 2] if init_dir < 2 else DIR_TO_VEC[init_dir - 2]
            return init_pos[0] + dir_vec[0]*num_steps == env.agent_pos[0] and init_pos[1] + dir_vec[1]*num_steps == env.agent_pos[1]
    elif phrase_key == "go_":
        def goal_check(env, act):
            dir_vec = DIR_TO_VEC[env.agent_dir]
            in_front = env.grid.get(env.agent_pos[0] + dir_vec[0], env.agent_pos[1] + dir_vec[1])
            return manhattan_distance(env.agent_pos, target_pos) == 1 and in_front == target_obj
    elif phrase_key == "pickup_":
        def goal_check(env, act):
            correct_distance_away = manhattan_distance(env.agent_pos, target_pos) == 1
            carrying = env.carrying is not None and env.carrying.color == target_obj.color and type(env.carrying) == type(target_obj)
            return correct_distance_away and carrying
    elif phrase_key == "put_":
        def goal_check(env, act):
            return env.carrying is None
    elif phrase_key == "open_" or phrase_key == "unlock":
        def goal_check(env, act):
            return target_obj.is_open
    elif phrase_key == "close_":
        def goal_check(env, act):
            return not target_obj.is_open
    return goal_check


def query_model_intention_speaker(mismatch, task, model, training_data, test_data, shot):
    assert shot in ["zero", "few"]
    if "3.5" in model:
        tpm_limit = 1000000
    else:
        tpm_limit = 8000000
    
    debug("Starting to generate outputs")
    grand_outputs = ""
    current_token_count = 0
    num_training_samples = len(training_data)

    for idx, row in tqdm(test_data.iterrows()):
        if shot == "few":
            sample_indices = np.random.choice(num_training_samples, size = 3, replace = False)
            traj_examples = []
            skill_examples = []
            for i in sample_indices:
                traj_examples.append(training_data.iloc[i]["traj_fully_obs_text"])
                skill_examples.append(training_data.iloc[i]["skill"])
            example_prefix = "Below are three examples of observation-action sequences and the correct skill name for them."
            joined_examples = "\n-----\n".join([f"EXAMPLE {i + 1}:\n{traj_examples[i]}\nCORRECT ANSWER: {skill_examples[i]}" for i in range(3)])
            few_shot_examples = f"{example_prefix}\n{joined_examples}"
            full_prompt = GET_SKILL_NAME_QUESTION_FEW_SHOT.format(few_shot_examples = few_shot_examples, obs_act_seq = row["traj_fully_obs_text"])
        else:
            full_prompt = GET_SKILL_NAME_QUESTION.format(obs_act_seq = row["traj_fully_obs_text"])
        
        tokens_needed = len(full_prompt) // 4
        current_token_count += tokens_needed
        if current_token_count > tpm_limit:
            time.sleep(60)
            current_token_count = tokens_needed
        
        messages = [{"role": "user", "content": full_prompt}]
        try:
            response_obj = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = TEMPERATURE
            )
            response = response_obj["choices"][0]["message"]["content"]
            grand_outputs += f"PROMPT {idx} IS: {full_prompt}"
            grand_outputs += f"ANSWER IS: {row['skill']}\n"
            grand_outputs += f"MODEL RESPONDED: {response}\n\n"
        except Exception as e:
            print("Could not complete LLM request due to", e)
    
    with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if shot == 'few' else ''}.txt", "w") as f:
        f.write(grand_outputs)
    debug("Finished generating and decoding")


def evaluate_model(model, mismatch, task, few_shot, test_data):
    if mismatch == "intention":
        if task == "speaker":
            evaluate_model_intention_speaker(model, mismatch, task, few_shot)
        else:
            evaluate_model_intention_listener(mismatch, task, test_data, few_shot)


def evaluate_model_intention_speaker(model, mismatch, task, few_shot):
    debug("Starting to evaluate model")
    with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if few_shot else ''}.txt", "r") as f:
        logs = f.readlines()
    if "3.5" in model:
        tpm_limit = 1000000
    else:
        tpm_limit = 8000000
    grand_outputs = ""
    current_token_count = 0
    matches = []
    batch_limit = 50
    batch = []
    curr_batch_len = 0

    def evaluate_batch(batch):
        nonlocal current_token_count
        nonlocal grand_outputs
        nonlocal matches

        pairs_list = "\n".join([f"{j + 1}. {pair[0]}, {pair[1]}" for j, pair in enumerate(batch)])
        evaluation_prompt = f"Below are {batch_limit} pairs of what we'll call 'skill names.' Please answer in a numbered list with 'Yes' or 'No' for each pair, denoting whether the comma-separated skill names in the pair mean the same thing or not. For example, 'right' and 'turn right' mean the same thing. 'Pickup_green_box' and 'procure_green_box' mean the same thing. On the other hand, 'go_to_blue_ball' and 'retrieve_blue_ball' do NOT mean the same thing, as the former is to just go to the ball while the latter is to actually obtain it/pick it up.\n\nTwo important notes: (1) Spaces vs. underscores do not make a difference. (2) Make sure that in order for two skill names to have the same meaning, they must also refer to the same colored object! So 'go_to_blue_ball' is NOT the same as 'go_to_brown_ball'. (3) A special case is that if the first skill name in the pair is something like 'toggle', 'switch', or 'activate', then as long as the second skill name in the pair involves someting like opening, closing, or unlocking, they can be considered as having the same meaning.\n\nThe list of pairs is below. Please format your answer as a numbered list and do not say anything else in your response other than this list of {batch_limit}.\n{pairs_list}"
        
        tokens_needed = len(evaluation_prompt) // 4
        current_token_count += tokens_needed
        if current_token_count > tpm_limit:
            time.sleep(60)
            current_token_count = tokens_needed
        
        messages = [{"role": "user", "content": evaluation_prompt}]
        try:
            response_obj = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = TEMPERATURE
            )
            response = response_obj["choices"][0]["message"]["content"]
            backwards_offset = len(batch) - 1
            for evaluation in response.split("\n"):
                if "yes" in evaluation.lower():
                    grand_outputs += f"PROMPT {curr_idx - backwards_offset}: match!!!\n"
                    matches.append(1)
                else:
                    grand_outputs += f"PROMPT {curr_idx - backwards_offset}: no match :(\n"
                    matches.append(0)
                backwards_offset -= 1
        except Exception as e:
            print("Could not complete LLM request due to", e)
    
    for line in tqdm(logs):
        if line.startswith("PROMPT"):
            curr_idx = int(re.search(r"PROMPT (\d+)", line).group(1))
        elif line.startswith("ANSWER"):
            batch.append([re.search(r"ANSWER IS: (.*)\n", line).group(1)])
        elif line.startswith("MODEL"):
            batch[-1].append(re.search(r"MODEL RESPONDED: (.*)\n", line).group(1))
            curr_batch_len += 1
        if curr_batch_len == batch_limit:
            evaluate_batch(batch)
            curr_batch_len = 0
            batch = []
    if len(batch) != 0:
        evaluate_batch(batch)
    
    with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if few_shot else ''}_evaluation.txt", "w") as f:
        f.write(grand_outputs)
        f.write(f"MODEL MATCH PERCENTAGE OUT OF {len(matches)}: {np.mean(matches)}")
    debug("Finished evaluating model")


def evaluate_model_intention_listener(mismatch, task, test_data, few_shot):
    debug("Starting to evaluate model")
    with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if few_shot else ''}.txt", "r") as f:
        logs = f.readlines()
    grand_outputs = ""
    matches = []

    def evaluate_trajectory(ground_truth_traj, model_traj):
        nonlocal grand_outputs
        nonlocal matches

        grand_outputs += f"CORRECT TRAJECTORY: {ground_truth_traj}\n"
        grand_outputs += f"MODEL TRAJECTORY: {model_traj}\n"
        
        if len(model_traj) >= 24:
            matches.append(0)
            grand_outputs += "Match percentage: 0.00\n\n"
        else:
            traj_match_tracker = 0
            for i in range(len(ground_truth_traj)):
                try:
                    if ground_truth_traj[i] == model_traj[i]:
                        traj_match_tracker += 1
                    else:
                        break
                except IndexError:  # model trajectory is too short. We don't care if it's too long because if it reaches that point then it already matched the rest of the ground truth trajectory and achieved the goal.
                    break
            match_pct = traj_match_tracker / len(ground_truth_traj)
            matches.append(match_pct)
            grand_outputs += f"Match percentage: {match_pct}\n\n"
    
    trajectory = []
    curr_traj_len = 0
    for line in tqdm(logs):
        if line.startswith("PROMPT"):
            if len(trajectory) != 0:
                evaluate_trajectory(test_data.iloc[curr_idx]["actions"], trajectory)
                curr_traj_len = 0
                trajectory = []
            curr_idx = int(re.search(r"PROMPT (\d+)", line).group(1))
            grand_outputs += f"PROMPT {curr_idx}:\n"
        elif line.startswith("MODEL"):
            trajectory.append(int(re.search(r"IS: (.*)\n", line).group(1)))
            curr_traj_len += 1
        elif line.startswith("SUCCESS"):
            curr_traj_len = 0
            trajectory = []
            matches.append(1)
            grand_outputs += f"CORRECT TRAJECTORY: {test_data.iloc[curr_idx]['actions']}\n"
            grand_outputs += "Match percentage: 1.00\n\n"
    if len(trajectory) != 0:
        evaluate_trajectory(test_data.iloc[curr_idx]["actions"], trajectory)
    
    with open(f"baselines/{mismatch}_{task}_llm_query_baseline{'_few_shot' if few_shot else ''}_evaluation.txt", "w") as f:
        f.write(grand_outputs)
        f.write(f"MODEL MATCH PERCENTAGE OUT OF {len(matches)}: {np.mean(matches)}")
    debug("Finished evaluating model")


def main(mismatch: str, task: str, evaluating: bool = False, few_shot: bool = False):
    training_data, _, test_data = get_datasets(mismatch)
    if evaluating:
        start = time.time()
        evaluate_model("gpt-4-turbo", mismatch, task, few_shot, test_data)
        end = time.time()
        debug(f"Evaluating {mismatch} {task} task took {format_seconds(end - start)}")
    else:
        start = time.time()
        query_model(mismatch, task, "gpt-4-turbo", training_data, test_data, "few" if few_shot else "zero")
        end = time.time()
        debug(f"Querying for {mismatch} {task} task took {format_seconds(end - start)}")
    

if __name__ == "__main__":
    continuous_logging = True

    parser = ArgumentParser()
    parser.add_argument("--belief", "-b", action = "store_true")
    parser.add_argument("--intention", "-i", action = "store_true")
    parser.add_argument("--speaker", "-s", action = "store_true")
    parser.add_argument("--listener", "-l", action = "store_true")
    parser.add_argument("--few_shot", "-f", action = "store_true")
    parser.add_argument("--evaluate", "-e", action = "store_true")
    args = parser.parse_args()

    assert xor(args.intention, args.belief, none_check = False), "Only one type of mismatch please!"
    assert xor(args.speaker, args.listener, none_check = False), "Only one type of task please!"

    main(
        "intention" if args.intention else "belief",
        "speaker" if args.speaker else "listener",
        args.evaluate,
        args.few_shot
    )
