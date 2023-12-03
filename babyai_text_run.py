import babyai_text
import gym
from minigrid.manual_control import ManualControl
from constants import *
import numpy as np
import random
from agent import *
from constants import *
from utils import *
import argparse
from envs import *
import pygame

max_msg_tokens = MAX_MSG_TOKENS
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

instruction_types = ["high", "low-teach", "low-learn", "mid-direct", "mid-explore", "mid-direct-explore", "mid-avoid", "mid-explore-avoid"]

def manual_test(env, seed):
    manual_control = ManualControl(env, seed = seed)
    manual_control.start()

def generate_env(scenario):
    if scenario == 1:
        """
        LA's environment is harder than TA's
        TA: "BabyAI-GoToRedBallNoDists-v0"
        LA: "BabyAI-GoToImpUnlock-v0"
        """
        learning_env = gym.make("BabyAI-GoToImpUnlock-v0")
    elif scenario == 2:
        """
        LA's environment is easier than TA's
        TA: "BabyAI-GoToImpUnlock-v0"
        LA: "BabyAI-GoToRedBallNoDists-v0"
        """
        learning_env = gym.make("BabyAI-GoToRedBallNoDists-v0")
    elif scenario == 3:
        """
        LA's environment is more confusing than TA's
        TA: "BabyAI-PickupDist-v0"
        LA: "BabyAI-PickupDist-v0"
        """
        learning_env = gym.make("BabyAI-PickupDist-v0")
    return learning_env

def construct_instruction(scenario: int, inst_type: str, env_obs: dict):
    if inst_type == "high":
        if scenario == 1 or scenario == 2:
            instruction = env_obs["mission"].capitalize() + "."
        elif scenario == 3:
            instruction = env_obs["mission"].capitalize().replace("a", "the") + "."
    elif inst_type == "low-teach":
        if scenario == 1:
            obj = " ".join(env_obs["mission"].split()[-2:])
            instruction = f"Move forward four steps. Turn left. Move forward one step to reach the {obj}."
        elif scenario == 2:
            instruction = "Turn right and move forward two steps. Open the blue door. Move forward four steps, turn left, move forward one step, turn right, and move forward two steps. Open the gray door. Move forward two steps and turn left. Pick up the purple key. Turn right. Move forward eight steps, going through the open gray door. Turn left, move forward one step, and turn right. Now move forward five steps, going through the open blue door, and turn right. Move forward four steps. Unlock the purple door with the purple key. Move forward two steps, turn left, then move forward two steps to reach the red ball."
        elif scenario == 3:
            obj_desc =  env_obs["mission"].split()[-2:]
            if obj_desc[1] == "object":  # [color] object
                pass  # TODO: finish this
            instruction = "Turn right. Move forward two steps. Pick up "
    return instruction

def manual_main(scenario, seed):
    la = LearningAgent("Gingerbread Man", "openai", "gpt-3.5-turbo")
    learning_env = generate_env(scenario)
    while True:
        teaching_input = input().split()
        if teaching_input[0] == "inst":  # give initial instruction
            obs, info = learning_env.reset(seed = seed)
            learning_env.render()
            learning_env.render()
            print("**********************************************************")
            # la.set_instruction(input())
            with open("instruction.txt", "r") as f:
                la.set_instruction(f.read())
            print(f"{la.name} was told: {la.system_message['content']}")
        elif teaching_input[0] == "obs":  # give agent the observation
            obs_desc = ". ".join(info["descriptions"]) + ". "
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "retry":  # try again to get a valid action
            obs_desc = ". ".join(info["descriptions"]) + ". "
            action = la.get_action(obs_desc, action_failed = True)
            print(f"{la.name} was told to retry: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "act":
            for primitive_action in teaching_input[1:]:
                obs, reward, done, info = learning_env.step(ACTION_TO_IDX[primitive_action])
            if reward > 0:
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
        learning_env.render()
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", "-s", type = int)
    parser.add_argument("--seed", "-d", type = int)
    args = parser.parse_args()
    manual_main(1, 100)
