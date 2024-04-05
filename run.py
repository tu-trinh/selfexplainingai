from package.infrastructure.basic_utils import xor, debug
from package.builder import make_agents
from package.game import Game
from package.enums import MessageType
from package.message import Message

import argparse
import time
import pickle


def yaml_builder(args):
    file_name = ""
    if args.belief_mismatch:
        file_name += "belief_"
    elif args.intention_mismatch:
        file_name += "intention_"
    elif args.reward_mismatch:
        file_name += "reward_"
    if args.speaker_task:
        file_name += "speaker_"
    elif args.listener_task:
        file_name += "listener_"
    file_name += f"difficulty{args.difficulty}.yaml"
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief_mismatch", "-b", action = "store_true")
    parser.add_argument("--intention_mismatch", "-i", action = "store_true")
    parser.add_argument("--reward_mismatch", "-r", action = "store_true")
    parser.add_argument("--speaker_task", "-s", action = "store_true")
    parser.add_argument("--listener_task", "-l", action = "store_true")
    parser.add_argument("--difficulty", "-d", type = int, required = True, choices = [1, 2, 3])
    args = parser.parse_args()

    assert xor(args.belief_mismatch, args.intention_mismatch, args.reward_mismatch, none_check = False), "Exactly one type of mismatch needed"
    assert xor(args.speaker_task, args.listener_task, none_check = False), "Exactly one type of task needed"

    principal, attendant = make_agents(f"./package/configs/{yaml_builder(args)}")
    p_env = principal.world_model
    p_env.render()
    time.sleep(3)
    debug("resetting")
    p_env.reset()
    p_env.render()
    time.sleep(3)
    debug("resetting")
    debug(p_env)
    _, info = p_env.reset(seed = 69)
    p_env = info["new_inst"]
    debug(p_env)
    p_env.render()
    time.sleep(3)
    # debug("Finding optimal solution took", round((e - s) / 60, 3), "minutes")
    # game = Game(principal, attendant)
    # game.run("i", "a")
    # game.evaluate("a", "p")
