from package.builder import *
from environment_play import *

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mismatch", "-m", type = str, required = True, choices = ["belief", "intention"])
    parser.add_argument("--side", "-s", type = str, required = True, choices = ["speaker", "listener"])
    parser.add_argument("--difficulty", "-d", type = int, required = True, choices = [1, 2, 3])
    args = parser.parse_args()

    principal, attendant = make_agents(f"./package/configs/{args.mismatch}_{args.side}_difficulty{args.difficulty}.yaml")
    env = principal.world_model
    time.sleep(5)
    debug("editing")
    env.hide_keys()
    debug("done")
    env.render()
    time.sleep(5)
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
