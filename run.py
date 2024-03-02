from package.infrastructure.basic_utils import xor, debug
from package.builder import make_agents
from package.game import Game

import argparse
import time


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
    # time.sleep(5)
    s = time.time()
    policy = attendant._find_optimal_policy()
    e = time.time()
    debug(policy)
    debug("Finding optimal solution took", round((e - s) / 60, 3), "minutes")
    # game = Game(principal, attendant)
    # game.run("i", "a")
    # game.evaluate("a", "p")
    # attendant.world_model.render()
    # time.sleep(5)
    # attendant.generate_skill_descriptions()
    # env = principal.world_model
    # env.render()
    # time.sleep(3)
    # apply_edits(env, [{"name": "change_target_color"}, {"name": "hide_keys"}, {"name": "add_opening_to_wall"}])
    # env.render()
    # time.sleep(3)
    # message = attendant.speak(Message(MessageType.INTENTION_START))
    # message = principal.listen(message)
    # message = principal.speak(Message(MessageType.INTENTION_START))
    # message = attendant.listen(message)
    """
    ## Speaking task
    if args.side == "speaker":
        # A: generates a skill description given a trajectory
        skill_summaries = attendant.speak(mode = "inform")
        print(skill_summaries)
        ### A: for each available skill (that is not part of the generalize to set), get obs-action pairs and then ask to summarize it into one skill. parse the response
        ### A: give this set of skills to P
        # P: generate a plan: sequence of those skills (in brackets?) for the given high level task, in language
        # language_plan = principal.listen(skills = skill_summaries, mode = "plan")
        # Different difficulty levels:
        ### Level 1 (env generalization): 
        ### Level 2 (skill generalization):
        ### Level 3 (both generalizations):
        # For now, just work on the feedback -> second plan generation pipeline. Then figure out first plan -> feedback.

    ## Listening task
    elif args.side == "listener":
        # P: generates skill descriptions (for now, just assume it can give function names)
        skill_descriptions = principal.speak(mode = "inform")
        # A: returns a language plan using the available P skills
        language_plan = attendant.speak(skills = skill_descriptions, mode = "plan")
        ### A: first figure out its own optimal trajectory plan (obs-action pairs)
        ### A: then convert it into an action plan for P
        # Different difficulty levels:
        ### Just one for now: P has fewer skills than A
    """
