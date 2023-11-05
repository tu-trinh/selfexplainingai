from minigrid.wrappers import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.manual_control import ManualControl
from mdp import GoalEnv
from model import MinigridFeaturesExtractor
import gymnasium as gym
from utils import *
import argparse
from agent import *
import numpy as np
import random
import time

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def manual_test(env):
    manual_control = ManualControl(env, seed = 42)
    manual_control.start()

def generate_env(scenario, env_num):
    if scenario == 1:
        """
        Scenario one: LA's environment is harder than TA's
        Teaching:
        S E E       S E E E E
        E E E       E E E E E
        E E G       E E E E E
                    E E E E E
                    E E E E G
        Learning:
        S W E       S E W E E
        E D E       E E W E E
        K W G       E E D E E
                    E E W E E
                    K E W E G
        """
        # teaching_env = GoalEnv(env_id = "empty_5x5", size = 7, goals = [((5, 5), "green")], walls = [], doors = [], keys = [], render_mode = "human", agent_view_size = 5)
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        goal_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
        if wall_orientation == "vertical":
            if goal_pos[0] > room_size // 2:
                wall_col = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([goal_pos[0]])))
                key_x_lb, key_x_ub = 1, wall_col
                key_y_lb, key_y_ub = 1, room_size - 1
            else:
                wall_col = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([goal_pos[0]])))
                key_x_lb, key_x_ub = wall_col + 1, room_size - 1
                key_y_lb, key_y_ub = 1, room_size - 1
            walls = [(wall_col, y) for y in range(1, room_size - 1)]
        elif wall_orientation == "horizontal":
            if goal_pos[1] > room_size // 2:
                wall_row = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([goal_pos[1]])))
                key_x_lb, key_x_ub = 1, room_size - 1
                key_y_lb, key_y_ub = 1, wall_row
            else:
                wall_row = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([goal_pos[1]])))
                key_x_lb, key_x_ub = 1, room_size - 1
                key_y_lb, key_y_ub = wall_row + 1, room_size - 1
            walls = [(x, wall_row) for x in range(1, room_size - 1)]
        key_pos = (np.random.randint(key_x_lb, key_x_ub), np.random.randint(key_y_lb, key_y_ub))
        door_pos = random.choice(walls)
        agent_pos = (np.random.randint(key_x_lb, key_x_ub), np.random.randint(key_y_lb, key_y_ub))
        while agent_pos in walls or agent_pos in [door_pos, key_pos, goal_pos]:
            agent_pos = (np.random.randint(key_x_lb, key_x_ub), np.random.randint(key_y_lb, key_y_ub))
        agent_dir = np.random.randint(0, 4)
        learning_env = GoalEnv(env_id = f"door_key_{room_size}_{env_num}", size = room_size,
                               agent_start_pos = agent_pos, agent_start_dir = agent_dir,
                               goals = [(goal_pos, "green")], walls = walls, doors = [(door_pos, "blue", True)],
                               keys = [(key_pos, "blue")], render_mode = "human", agent_view_size = AGENT_VIEW_SIZE)
    elif scenario == 2:
        """
        Scenario two: LA's environment is easier than TA's
        Teaching:
        S W E       S E W E E
        E D E       E E W E E
        K W G       E E D E E
                    E E W E E
                    K E W E E
        Learning:
        S E E       S E E E E
        E E E       E E E E E
        E E G       E E E E E
                    E E E E E
                    E E E E G
        """
        # teaching_env = GoalEnv(env_id = "door_key_5x5", size = 7, goals = [((5, 5), "green")], walls = [(3, y) for y in range(1, 6)], doors = [((3, 3), "blue", True)], keys = [((1, 5), "blue")], render_mode = "human", agent_view_size = 5)
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        goal_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        while agent_pos == goal_pos:
            agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        agent_dir = np.random.randint(0, 4)
        learning_env = GoalEnv(env_id = f"empty_{room_size}_{env_num}", size = room_size, agent_start_pos = agent_pos,
                               agent_start_dir = agent_dir, goals = [(goal_pos, "green")], walls = [], doors = [],
                               keys = [], render_mode = "human", agent_view_size = AGENT_VIEW_SIZE)
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
    elif scenario == 4:
        """
        Scenario four: LA has different capabilities from TA
        Teaching:
        S D G       S E D E G
        E W E       E E W E E
        E E E       E E W E E
                    E E E E E
                    E E E E E
        Learning:
        S D G       S E D E G
        E W E       E E W E E
        E E E       E E W E E
                    E E E E E
                    E E E E E
        """
        # teaching_env = GoalEnv(env_id = "door_5x5", size = 7, goals = [((5, 1), "green")], walls = [(3, 1), (3, 2), (3, 3)], doors = [((3, 1), "purple", False)], keys = [], render_mode = "human", agent_view_size = 5)
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        goal_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        while agent_pos == goal_pos or abs(agent_pos[0] - goal_pos[0]) <= 1 or abs(agent_pos[1] - goal_pos[1]) <= 1:
            agent_pos = (np.random.randint(1, room_size - 1), np.random.randint(1, room_size - 1))
        agent_dir = np.random.randint(0, 4)
        if agent_pos[0] == goal_pos[0]:
            wall_orientation = "horizontal"
        elif agent_pos[1] == goal_pos[1]:
            wall_orientation = "vertical"
        else:
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
        wall_start = "head" if np.random.random() > 0.5 else "tail"
        if wall_orientation == "vertical":
            if agent_pos[0] < goal_pos[0]:
                left, right = agent_pos[0] + 1, goal_pos[0]
            else:
                left, right = goal_pos[0] + 1, agent_pos[0]
            wall_col = np.random.randint(left, right)
            if wall_start == "head":
                walls = [(wall_col, y) for y in range(1, room_size // 2 + 1)]
            elif wall_start == "tail":
                walls = [(wall_col, y) for y in range(room_size // 2 - 1, room_size - 1)]
        elif wall_orientation == "horizontal":
            if agent_pos[1] < goal_pos[1]:
                top, bottom = agent_pos[1] + 1, goal_pos[1]
            else:
                top, bottom = goal_pos[1] + 1, agent_pos[1]
            wall_row = np.random.randint(top, bottom)
            if wall_start == "head":
                walls = [(x, wall_row) for x in range(1, room_size // 2 + 1)]
            elif wall_start == "tail":
                walls = [(x, wall_row) for x in range(room_size // 2 - 1, room_size - 1)]
        door_pos = random.choice(walls)
        learning_env = GoalEnv(env_id = f"door_wall_{room_size}_{env_num}", size = room_size, agent_start_pos = agent_pos, agent_start_dir = agent_dir,
                               goals = [(goal_pos, "green")], walls = walls, doors = [(door_pos, "purple", True)],
                               keys = [], render_mode = "human", agent_view_size = AGENT_VIEW_SIZE)
    return learning_env


def main(scenario, instruction, inst_type):
    la = LearningAgent("Bartolomé de las Casas", "openai", "gpt-3.5-turbo")
    for i in range(100):
        try:
            learning_env = generate_env(scenario, i)
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
                        action_choice = int(agent_resp.split(", ")[0])
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
            with open(f"scenario{scenario}_type{inst_type}.csv", "a") as f:
                steps_to_succeed = total_steps if success else float("inf")
                f.write(f"{scenario},{learning_env.env_id},{inst_type},{instruction},{success},{steps_to_succeed},{la.interactions}\n")
        except Exception as e:
            print(f"Scenario {scenario} with instruction type {inst_type} failed on environment {i} with the following exception: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", type = int, required = True)
    parser.add_argument("-t", "--inst_type", type = int, required = True)
    args = parser.parse_args()
    # manual_test(learning_env)
    instruction_types = ["high", "low", "mid-direct", "mid-explore", "mid-direct-explore", "mid-avoid", "mid-explore-avoid"]
    experiment_mapping = {
        1: {
            "high": "Get to the goal.",
            "low": "Go forward all the way until you hit a wall, then turn right, then go forward all the way again until you hit the goal.",
            "mid-direct": "Find and pick up a key, use it to unlock the door, and go through the door to get to the goal.",
            "mid-explore": "Get to the goal. If you don't see the goal at first, explore the room more until you see it.",
            "mid-direct-explore": "Pick up the key, use it to unlock the door, and go through the door to get to the goal. For each of those three items, if you don't see it first, explore the room until you find it."
            "mid-avoid": "Get to the goal. You may need to remove some obstacles in order to see and reach this goal.",
            "mid-explore-avoid": "Get to the goal. You may need to remove some obstacles in order to see and reach this goal. Always look around in case the goal is nearby."
        },
        2: {
            "high": "Get to the goal.",
            "low": "Turn right and move forward until you see a key and pick it up. Then turn around, go forward one step, and turn right to use the key to unlock the door. Then go through the door and continue walking until you hit a wall. Then turn right and go forward until you hit the goal.",
            "mid-direct": "There are no obstacles around you. Find the goal and go straight to it.",
            "mid-explore": "Get to the goal. If you don’t see the goal at first, try exploring the room more until you find it.",
            "mid-direct-explore": "There are no obstacles aorund you. Find the goal and go straight to it. If you don't see it at first, explore the room more until you find it.",
            "mid-avoid": "Get to the goal by walking around and seeing if you need to remove any obstacles or if you can see and reach the goal directly.",
            "mid-explore-avoid": "Get to the goal by walking around and seeing if you need to remove obstacles first. If you don’t see the goal, try exploring around more until you find it."
        },
        3: {
            "high": "Get to the goal.",
            "low": "Walk forward until you hit a wall, then turn and keep walking forward until you hit the goal.",
            "mid-direct": "There are no obstacles around you. Go straight to the green goal, not another color goal.",
            "mid-explore": "Get to the green goal. If you don’t see it anywhere, explore the room more until you find it.",
            "mid-direct-explore": "Go straight to the green goal, not another color goal. If you don’t see it anywhere, explore the room more until you find it.",
            "mid-avoid": "Get to the green goal. Avoid everything else.",
            "mid-explore-avoid": "Get to the green goal and avoid everything else. If you don't see it at first, explore the room more until you find it."
        },
        4: {
            "high": "Get to the goal.",
            "low": "Walk forward through the open door. Then keep walking forward until you get to the goal.",
            "mid-direct": "Get to the goal by walking around the wall. There is no key to open the door for a shortcut.",
            "mid-explore": "Get to the goal. If you don’t see it anywhere, explore the room more until you find it.",
            "mid-direct-explore": "Get to the goal. If you get stuck at the door because there is no key, go around the wall. If you do not see the goal at first, explore the room until you find it.",
            "mid-avoid": "Get to the green goal. If you cannot go through the door or wall, try finding a different way around them.",
            "mid-explore-avoid": "Get to the green goal. If you cannot go through the door or wall, try finding a different way around them. If you do not see the goal at first, keep exploring the room until you find it."
        }
    }
    main(args.scenario,
         experiment_mapping[args.scenario][instruction_types[args.inst_type]],
         instruction_types[args.inst_type])
