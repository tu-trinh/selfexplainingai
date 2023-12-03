from minigrid.wrappers import *
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
from minigrid.manual_control import ManualControl
from envs import GoalEnv
# from model import MinigridFeaturesExtractor
import gymnasium
import gym
from utils import *
import argparse
from agent import *
import numpy as np
import random
import time
import copy
import re
from constants import *

max_msg_tokens = MAX_MSG_TOKENS


instruction_types = ["high", "low", "mid-direct", "mid-explore", "mid-direct-explore", "mid-avoid", "mid-explore-avoid"]

def manual_test(env, seed):
    manual_control = ManualControl(env, seed = seed)
    manual_control.start()

def generate_env(scenario, env_num):
    elif scenario == 3:
        """
        Scenario three: LA's environment is more confusing than TA's
        Teaching:
        S E E       S E E E E
        E E E       E E E E E
        E E Gg      E E E E E
                    E E E E E
                    E E E E Gg
        Learning:
        S E Gr      S E E E Gr
        E E E       E E E E E
        E E Gg      E E E E E
                    E E E E E
                    E E E E Gg
        """
        # teaching_env = GoalEnv(env_id = "empty_5x5", size = 7, goals = [((5, 5), "green")], walls = [], doors = [], keys = [], render_mode = "human", agent_view_size = 5)
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        num_goals = np.random.randint(2, 6)
        goal_positions = []
        for _ in range(num_goals):
            goal_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
            while goal_pos in goal_positions:
                goal_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
            goal_positions.append(goal_pos)
        agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        while agent_pos in goal_positions:
            agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        agent_dir = np.random.randint(0, 4)
        learning_env = GoalEnv(env_id = f"multiple_goals_{room_size}_{env_num}", size = room_size,
                               agent_start_pos = agent_pos, agent_start_dir = agent_dir,
                               goals = [(goal_positions[i], GOAL_COLOR_NAMES[i]) for i in range(num_goals)],
                               walls = [], doors = [], keys = [], render_mode = "human",
                               agent_view_size = AGENT_VIEW_SIZE)
    return learning_env


def automated_main(scenario, instruction, inst_type, start_idx = 0, end_idx = 100):
    la = LearningAgent("Bartolomé de las Casas", "openai", "gpt-3.5-turbo")
    iter_tries = {i: 0 for i in range(start_idx, end_idx)}
    i = start_idx
    retry_env = None
    while i < end_idx:
        try:
            if iter_tries[i] == 0:
                learning_env = generate_env(scenario, i)
                retry_env = copy.deepcopy(learning_env)
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
                    print(f"STARTING ENVIRONMENT {i}")
                    la.set_instruction(instruction)
                    print(f"{la.name} was told: {la.system_message['content']}\n")
                    in_progress = True
                    obs, _ = learning_env.reset()
                obs_desc = get_obs_desc(obs)
                while tries < MAX_TRIES:
                    if tries == 0:
                        agent_resp = la.get_action(obs_desc, action_failed = action_failed)
                        print(f"{la.name} was told: {la.prompts[-1]}")
                    else:
                        agent_resp = la.get_action()
                        print(f"{la.name} was asked to try again.")
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

def manual_main(scenario, seed):
    la = LearningAgent("Gingerbread Man", "openai", "gpt-3.5-turbo")
    learning_env = generate_env(scenario, seed)
    while True:
        teaching_input = input().split()
        if teaching_input[0] == "inst":  # give initial instruction
            obs, _ = learning_env.reset()
            print("**********************************************************")
            with open("instruction.txt", "r") as f:
                la.set_instruction(f.read(), list_available_actions(scenario))
            print(f"{la.name} was told: {la.system_message['content']}")
        elif teaching_input[0] == "obs":  # give agent the observation
            obs_desc = get_obs_desc(obs, detail = 1)
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "explore":  # give agent modified, "explored" observation
            learning_env.step(0)  # turning around
            backwards_obs, _, _, _, _ = learning_env.step(0)
            learning_env.step(0)  # turning back
            obs, _, _, _, _ = learning_env.step(0)
            obs_desc = get_obs_desc(obs, backwards_obs = backwards_obs, detail = 2)
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "retry":  # try again to get a valid action
            obs_desc = get_obs_desc(obs, detail = 1)
            action = la.get_action(obs_desc, action_failed = True)
            print(f"{la.name} was told to retry: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "act":
            for primitive_action in teaching_input[1:]:
                obs, _, _, _, _ = learning_env.step(ACTION_TO_IDX[primitive_action])
        elif teaching_input[0] == "succeed":
            print(f"{la.name} has succeeded! This round took {la.interactions} interactions.")
        elif teaching_input[0] == "end":
            print(f"{la.name} has failed to complete the task. This took {la.interactions} interactions.")
        elif teaching_input[0] == "hist":
            history = la.display_history()
            print(f"Previous interactions included the following:")
            for hist in history:
                print(hist)
        elif teaching_input[0] == "view":
            print("Current observation:")
            print(obs)
        print()


if __name__ == "__main__":
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    manual_main(1, seed)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--scenario", type = int, required = True)
    # parser.add_argument("-t", "--inst_type", type = int, required = True)
    # parser.add_argument("-si", "--start_idx", type = int, default = 0)
    # parser.add_argument("-ei", "--end_idx", type = int, default = 100)
    # args = parser.parse_args()

    # experiment_mapping = {
    #     1: {
    #         "high": "Get to the goal.",
    #         "low": "Go forward all the way until you hit a wall, then turn right, then go forward all the way again until you hit the goal.",
    #         "mid-direct": "Find and pick up a key, use it to unlock the door, and go through the door to get to the goal.",
    #         "mid-explore": "Get to the goal. If you don't see the goal at first, explore the room more until you see it.",
    #         "mid-direct-explore": "Pick up the key, use it to unlock the door, and go through the door to get to the goal. For each of the key, door, and goal, if you don't see it first, explore the room until you find it.",
    #         "mid-avoid": "Get to the goal. You may need to remove some obstacles in order to see and reach this goal.",
    #         "mid-explore-avoid": "Get to the goal. You may need to remove some obstacles in order to see and reach this goal. Always look around in case the goal is nearby."
    #     },
    #     2: {
    #         "high": "Get to the goal.",
    #         "low": "Turn right and move forward until you see a key and pick it up. Then turn around, go forward one step, and turn right to use the key to unlock the door. Then go through the door and continue walking until you hit a wall. Then turn right and go forward until you hit the goal.",
    #         "mid-direct": "There are no obstacles around you. Find the goal and go straight to it.",
    #         "mid-explore": "Get to the goal. If you don’t see the goal at first, try exploring the room more until you find it.",
    #         "mid-direct-explore": "There are no obstacles around you. Find the goal and go straight to it. If you don't see it at first, explore the room more until you find it.",
    #         "mid-avoid": "Get to the goal by walking around and seeing if you need to remove any obstacles or if you can see and reach the goal directly.",
    #         "mid-explore-avoid": "Get to the goal by walking around and seeing if you need to remove obstacles first. If you don’t see the goal, try exploring around more until you find it."
    #     },
    #     3: {
    #         "high": "Get to the goal.",
    #         "low": "Walk forward until you hit a wall, then turn and keep walking forward until you hit the goal.",
    #         "mid-direct": "There are no obstacles around you. Go straight to the green goal, not another color goal.",
    #         "mid-explore": "Get to the green goal. If you don't see it anywhere, explore the room more until you find it.",
    #         "mid-direct-explore": "Go straight to the green goal. If you don't see it anywhere, explore the room more until you find it. Do not go to a goal of another color.",
    #         "mid-avoid": "Get to the green goal. Avoid everything else.",
    #         "mid-explore-avoid": "Get to the green goal and avoid everything else. If you don't see it at first, explore the room more until you find it."
    #     },
    #     4: {
    #         "high": "Get to the goal.",
    #         "low": "Walk forward through the open door. Then keep walking forward until you get to the goal.",
    #         "mid-direct": "Get to the goal by walking around the wall. There is no key to open the door for a shortcut.",
    #         "mid-explore": "Get to the goal. If you don’t see it anywhere, explore the room more until you find it.",
    #         "mid-direct-explore": "Get to the goal. If you get stuck at the door because there is no key, go around the wall. If you do not see the goal at first, explore the room until you find it.",
    #         "mid-avoid": "Get to the green goal. If you cannot go through the door or wall, try finding a different way around them.",
    #         "mid-explore-avoid": "Get to the green goal. If you cannot go through the door or wall, try finding a different way around them. If you do not see the goal at first, keep exploring the room until you find it."
    #     }
    # }
    # start = time.time()
    # main(scenario = args.scenario,
    #      instruction = experiment_mapping[args.scenario][instruction_types[args.inst_type]],
    #      inst_type = instruction_types[args.inst_type],
    #      start_idx = args.start_idx,
    #      end_idx = args.end_idx)
    # end = time.time()
    # print(f"This took {format_seconds(end - start)} to run.")
