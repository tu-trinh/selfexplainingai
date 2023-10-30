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

door_key_COLOR_NAMES = ["blue", "purple", "yellow", "red", "green"]
goal_COLOR_NAMES = ["green", "red", "yellow", "purple", "blue"]

def manual_test(env):
    manual_control = ManualControl(env, seed = 42)
    manual_control.start()

def train(env):
    # policy_kwargs = dict(
    #     features_extractor_class = MinigridFeaturesExtractor,
    #     features_extractor_kwargs = dict(features_dim = 128),
    # )
    model = PPO("MultiInputPolicy", env, verbose = 1)
    model.learn(5000, progress_bar = True)
    model.save("ppo_multi_{}".format(env.env_id))

def test(env):
    model = PPO.load("ppo_{}".format(env_id))
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)

def view(env):
    env.reset()
    while True:
        action = int(input())
        obs, _, _, _, _ = env.step(action)
        print("Took action:", action)
        print(get_obs_desc(obs))

def main(la, learning_env, difficulty):
    # TODO: figure out how to include in obs what object it has in possession.
    while True:
        teaching_input = input().split()
        if teaching_input[0] == "inst":  # give initial instruction
            obs, _ = learning_env.reset()
            print("**********************************************************")
            la.set_instruction(input())
            print(f"{la.name} was told: {la.system_message['content']}")
        elif teaching_input[0] == "obs":  # give agent the observation
            obs_desc = get_obs_desc(obs, difficulty = difficulty)
            action = la.get_action(obs_desc)
            print(f"{la.name} was told: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "retry":  # try again to get a valid action
            action = la.get_action()
            print(f"{la.name} was again told: {la.prompts[-1]}")
            print(f"{la.name} says: {action}")
        elif teaching_input[0] == "act":
            for primitive_action in teaching_input[1:]:
                obs, reward, done, truncated, info = learning_env.step(ACTION_TO_IDX[primitive_action])
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
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", type = int, required = True)
    parser.add_argument("-d", "--difficulty", type = int, required = True)
    args = parser.parse_args()
    if args.scenario == 1:
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
        if args.difficulty == 3:
            teaching_env = GoalEnv(env_id = "empty_3x3", size = 5, goals = [((3, 3), "green")], walls = [], doors = [], keys = [],
                                render_mode = "human", agent_view_size = 3)
            learning_env = GoalEnv(env_id = "door_key_3x3", size = 5, goals = [((3, 3), "green")],
                                walls = [(2, y) for y in range(1, 4)], doors = [((2, 2), "blue", True)],
                                keys = [((1, 3), "blue")], render_mode = "human", agent_view_size = 3)
        elif args.difficulty == 5:
            teaching_env = GoalEnv(env_id = "empty_5x5", size = 7, goals = [((5, 5), "green")], walls = [], doors = [], keys = [],
                                   render_mode = "human", agent_view_size = 5)
            learning_env = GoalEnv(env_id = "door_key_5x5", size = 7, goals = [((5, 5), "green")],
                                   walls = [(3, y) for y in range(1, 6)], doors = [((3, 3), "blue", True)],
                                   keys = [((1, 5), "blue")], render_mode = "human", agent_view_size = 5)
    elif args.scenario == 2:
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
        if args.difficulty == 3:
            teaching_env = GoalEnv(env_id = "door_key_3x3", size = 5, goals = [((3, 3), "green")],
                                walls = [(2, y) for y in range(1, 4)], doors = [((2, 2), "blue", True)],
                                keys = [((1, 3), "blue")], render_mode = "human", agent_view_size = 3)
            learning_env = GoalEnv(env_id = "empty_3x3", size = 5, goals = [((3, 3), "green")], walls = [], doors = [], keys = [],
                                render_mode = "human", agent_view_size = 3)
        elif args.difficulty == 5:
            teaching_env = GoalEnv(env_id = "door_key_5x5", size = 7, goals = [((5, 5), "green")],
                                   walls = [(3, y) for y in range(1, 6)], doors = [((3, 3), "blue", True)],
                                   keys = [((1, 5), "blue")], render_mode = "human", agent_view_size = 5)
            learning_env = GoalEnv(env_id = "empty_5x5", size = 7, goals = [((5, 5), "green")], walls = [], doors = [], keys = [],
                                   render_mode = "human", agent_view_size = 5)
    elif args.scenario == 3:
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
        if args.difficulty == 3:
            teaching_env = GoalEnv(env_id = "empty_3x3", size = 5, goals = [((3, 3), "green")], walls = [], doors = [], keys = [],
                                render_mode = "human", agent_view_size = 3)
            learning_env = GoalEnv(env_id = "two_goals_3x3", size = 5, goals = [((3, 3), "green"), ((3, 1), "red")],
                                walls = [], doors = [], keys = [], render_mode = "human", agent_view_size = 3)
        elif args.difficulty == 5:
            teaching_env = GoalEnv(env_id = "empty_5x5", size = 7, goals = [((5, 5), "green")], walls = [], doors = [], keys = [],
                                   render_mode = "human", agent_view_size = 5)
            learning_env = GoalEnv(env_id = "two_goals_5x5", size = 7, goals = [((5, 5), "green"), ((5, 1), "red")], walls = [],
                                   doors = [], keys = [], render_mode = "human", agent_view_size = 5)
    elif args.scenario == 4:
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
        if args.difficulty == 3:
            teaching_env = GoalEnv(env_id = "door_3x3", size = 5, goals = [((3, 1), "green")], walls = [(2, 1), (2, 2)],
                                doors = [((2, 1), "purple", False)], keys = [], render_mode = "human", agent_view_size = 3)
            learning_env = GoalEnv(env_id = "door_3x3", size = 5, goals = [((3, 1), "green")], walls = [(2, 1), (2, 2)],
                                doors = [((2, 1), "purple", False)], keys = [], render_mode = "human", agent_view_size = 3)
        elif args.difficulty == 5:
            teaching_env = GoalEnv(env_id = "door_5x5", size = 7, goals = [((5, 1), "green")], walls = [(3, 1), (3, 2), (3, 3)],
                                doors = [((3, 1), "purple", False)], keys = [], render_mode = "human", agent_view_size = 5)
            learning_env = GoalEnv(env_id = "door_5x5", size = 7, goals = [((5, 1), "green")], walls = [(3, 1), (3, 2), (3, 3)],
                                doors = [((3, 1), "purple", False)], keys = [], render_mode = "human", agent_view_size = 5)
    
    la = LearningAgent("Fiona", "openai", "gpt-3.5-turbo")
    # manual_test(learning_env)
    main(la, learning_env, args.difficulty)