from package.infrastructure.llm_constants import MAX_MSG_TOKENS

from minigrid.manual_control import ManualControl

import numpy as np
import pandas as pd
import time
import copy
import re


max_msg_tokens = MAX_MSG_TOKENS
instruction_types = ["high", "low_teacher", "low_student", "mid_direct", "mid_explore", "mid_direct_explore",
                     "mid_avoid", "mid_explore_avoid"]


def manual_test(env, seed):
    manual_control = ManualControl(env, seed = seed)
    manual_control.start()


def automated_main(scenario, instruction, inst_type, start_idx = 0, end_idx = 10):
    la = Attendant("", "scale", "mistral")
    iter_tries = {i: 0 for i in range(start_idx, end_idx)}
    i = start_idx
    retry_env = None
    while i < end_idx:
        try:
            if iter_tries[i] == 0:
                teacher_env, student_env = make_envs()
                retry_env = copy.deepcopy(student_env)
            else:
                learning_env = retry_env
            success = False
            in_progress = False
            tries = 0
            total_steps = 0
            reward = 0
            prev_obs = None
            action_failed = False
            while not success:
                if not in_progress:
                    print("**********************************************************")
                    la.set_instruction(instruction)
                    print(f"{la.name} was told: {la.llm.system_message}\n")
                    in_progress = True
                    obs, _ = student_env.reset()
                obs_desc = get_obs_desc(obs)
                while tries < MAX_TRIES:
                    if tries == 0:
                        agent_resp = la.get_action(obs_desc, action_failed = action_failed)
                        print(f"{la.name} was told: {la.llm.prompts[-1]}\n")
                    else:
                        agent_resp = la.get_action()
                        print(f"{la.name} was asked to try again.\n")
                    tries += 1
                    print(f"{la.name} says: {agent_resp}\n")
                    if "stumped" not in agent_resp:
                        try:
                            action_choice = int(agent_resp.split(", ")[0])
                        except:  # stupid robot can't follow instructions
                            action_choice = convert_response_to_action(agent_resp)
                            if not action_choice:
                                raise Exception("Could not parse agent's response")
                        obs, reward, done, truncated, info = learning_env.step(CUSTOM_ACTION_TO_TRUE_ACTION[action_choice])
                        tries = 0
                        total_steps += 1
                        try:
                            np.testing.assert_equal(obs, prev_obs)
                            action_failed = True
                        except AssertionError:
                            action_failed = False
                        prev_obs = obs
                        break
                if reward > 0:
                    if learning_env.agent_pos == learning_env.goals[0][0]:  # if multiple goals, must reach green
                        print(f"{la.name} has succeeded! This round took {la.interactions} interactions.")
                        success = True
                    else:
                        reward = 0
                elif tries == MAX_TRIES:
                    print(f"{la.name} has failed to complete the task. This took {la.interactions} interactions.")
                    break
            with open(f"scenario{scenario}_type{instruction_types.index(inst_type)}.csv", "a") as f:
                steps_to_succeed = total_steps if success else float("inf")
                f.write(f"{scenario},{learning_env.env_id},{inst_type},{instruction},{success},{steps_to_succeed},{la.interactions}\n")
            i += 1
        except Exception as e:
            try_again_msg = "Trying again in one minute." if iter_tries[i] == 0 and "agent's response" not in str(e) else "Won't try again."
            print(f"Scenario {scenario} with instruction type {inst_type} failed on environment {i} with the following exception: {e}. {try_again_msg}")
            if iter_tries[i] > 0 or "agent's response" in str(e):
                i += 1
            else:
                time.sleep(60)
                iter_tries[i] += 1
                if "maximum context length" in str(e):
                    match = re.search(r"However, your messages resulted in (\d+) tokens", str(e))
                    diff = int(match.group(1)) - CONTEXT_WINDOWS[la.model]
                    max_msg_tokens -= diff


def manual_main(query_src, model_src, seed, inst_type, task, teacher_level, student_level = None, student_variants = None):
    la = Attendant("Geronimo Stilton", query_src, model_src)
    ref = pd.read_csv("./data/reference/reference.csv")
    _, student_env = make_envs(task = task,
                               teacher_level = teacher_level,
                               student_level = student_level,
                               student_variants = student_variants,
                               seed = seed)
    while True:
        teaching_input = input().split()
        if teaching_input[0] == "inst":  # give initial instruction
            obs, _ = student_env.reset()
            print("**********************************************************")
            print(f"STARTING ENVIRONMENT WITH SEED {seed}, INSTRUCTION {inst_type}")
            if student_variants and student_level:
                student_variants = [sv.value for sv in student_variants]
                relevant = ref[(ref["task"] == task.value) & (ref["teacher_level"] == teacher_level.value)
                               & (ref["student_level"] == student_level.value) & (ref["student_variants"] == student_variants)
                               & (ref["seed"] == seed)]
            elif student_level:
                relevant = ref[(ref["task"] == task.value) & (ref["teacher_level"] == teacher_level.value)
                               & (ref["student_level"] == student_level.value) & (ref["seed"] == seed)]
            elif student_variants:
                student_variants = [sv.value for sv in student_variants]
                relevant = ref[(ref["task"] == task.value) & (ref["teacher_level"] == teacher_level.value)
                               & (ref["student_variants"] == student_variants) & (ref["seed"] == seed)]
            assert len(relevant) > 0, "No corresponding row in the reference csv"
            instruction = relevant[f"inst_high"].iloc[0]
            if inst_type != "high":
                instruction += INSTRUCTION_PROLOGUE + relevant[f"inst_{inst_type}"].iloc[0]
            la.set_instruction(instruction)
            print(f"{la.name} was told: {la.llm.system_message}")
        
        elif teaching_input[0] == "obs":  # give agent the observation
            obs_desc = get_obs_desc(obs, detail = 1)
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.llm.prompts[-1]}")
            print(f"{la.name} says: {action}")
        
        elif teaching_input[0] == "explore":  # give agent modified, "explored" observation
            left_obs, _, _, _, _ = student_env.step(0)  # turning around
            backwards_obs, _, _, _, _ = student_env.step(0)
            right_obs, _, _, _, _ = student_env.step(0)  # turning back
            obs, _, _, _, _ = student_env.step(0)
            obs_desc = get_obs_desc(obs, left_obs = left_obs, backwards_obs = backwards_obs, right_obs = right_obs, detail = 2)
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.llm.prompts[-1]}")
            print(f"{la.name} says: {action}")
        
        elif teaching_input[0] == "redo":  # try again to get a valid action since previous action failed
            obs_desc = get_obs_desc(obs, detail = 1)
            action = la.get_action(obs_desc, action_failed = True)
            print(f"{la.name} was told to redo since the action failed: {la.llm.prompts[-1]}")
            print(f"{la.name} says: {action}")
        
        elif teaching_input[0] == "retry":  # try again if the agent was stumped to see if it has a better idea
            action = la.get_action()
            print(f"{la.name} was told to retry since it was stumped: {la.llm.prompts[-1]}")
            print(f"{la.name} says: {action}")
        
        elif teaching_input[0] == "act":
            for primitive_action in teaching_input[1:]:
                obs, _, _, _, _ = student_env.step(ACTION_TO_IDX[primitive_action])
        
        elif teaching_input[0] == "succeed":
            print(f"{la.name} has succeeded! This round took {la.interactions} interactions and {la.tokens} prompt tokens.")
        
        elif teaching_input[0] == "fail":
            print(f"{la.name} has failed to complete the task. This took {la.interactions} interactions and {la.tokens} prompt tokens.")
        
        print()
