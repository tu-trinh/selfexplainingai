from package.infrastructure.basic_utils import xor, debug
from package.builder import make_agents
from package.game import Game
from package.enums import MessageType
from package.message import Message
from environment_play import manual_test
from package.infrastructure.access_tokens import *
from package.infrastructure.llm_constants import *

import argparse
import time
import pickle
import requests


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
    # p, a = make_agents("./package/configs/intention_speaker_difficulty1.yaml")
    # env = p.world_model
    # manual_test(env, env.env_seed)


    parser = argparse.ArgumentParser()
    parser.add_argument("--belief_mismatch", "-b", action = "store_true")
    parser.add_argument("--intention_mismatch", "-i", action = "store_true")
    parser.add_argument("--reward_mismatch", "-r", action = "store_true")
    parser.add_argument("--speaker_task", "-s", action = "store_true")
    parser.add_argument("--listener_task", "-l", action = "store_true")
    parser.add_argument("--difficulty", "-d", type = int, required = True, choices = [1, 2, 3, 4])
    args = parser.parse_args()

    assert xor(args.belief_mismatch, args.intention_mismatch, args.reward_mismatch, none_check = False), "Exactly one type of mismatch needed"
    assert xor(args.speaker_task, args.listener_task, none_check = False), "Exactly one type of task needed"

    principal, attendant = make_agents(f"./package/configs/{yaml_builder(args)}")
    attendant.world_model.render_mode = "human"
    attendant.world_model.render()
    time.sleep(10)
    
    # for skill in attendant.skills:
    #     debug("Current skill:", skill)
    #     setup_actions, actions = attendant._retrieve_actions_from_skill_func(skill)
    #     obs_act_seq = attendant._generate_obs_act_sequence(actions, setup_actions)
    #     if "go_" in skill or "move_" in skill:
    #         hint = "This task has something to do with movement."
    #     else:
    #         hint = "This task has something to do with interacting with some object."
    #     prompt = f"There is a grid-like environment in which there is an AI agent. Some special quirks about this environment: locked doors can only be unlocked by keys of the same color as the door. In addition, when boxes are opened, they disappear and whatever was inside them replaces their spot on the grid. Now, we have an AI agent who is repeatedly given observations (environment descriptions) and then chooses actions to execute in response to the observations. This process has resulted in the below observation-and-action sequence. The ultimate goal of the AI agent in doing this sequence is to complete a task/skill/ability. Given this sequence, what do you think the AI agent is trying to do? (Hint: {hint}) In your response, on the first line please give me the task/skill/ability you think the agent is trying to do, then on subsequent lines give me 10 paraphrases of this task/skill/ability. Do not say anything else or format your response in any other way. Observation-and-action sequence below:\n{obs_act_seq}\n"
    #     debug("Asked the LLM")
    #     debug(prompt)
    #     url = f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    #     headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    #     payload = {
    #         "inputs": prompt,
    #         "options": {"wait_for_model": True},
    #         "parameters": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}
    #     }
    #     try:
    #         response_obj = requests.post(url, headers = headers, json = payload)
    #         # debug("LLM SAID?")
    #         # debug(response_obj.json())
    #         response = response_obj.json()[0]["generated_text"][len(prompt):]
    #         debug("LLM SAID!")
    #         debug(response)
    #     except Exception as e:
    #         print("Could not complete LLM request due to", e)
    #     print("\n\n\n")
