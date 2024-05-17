import pickle
from mindgrid.skills import Skills
from mindgrid.envs.edits import Edits
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.builder import make_env

def skillset_tasks():
    game_path = "datasets/skillset_listen_games_1000.pickle"
    data_path = "datasets/skillset_listen_data_1000.pickle"

    with open(game_path, "rb") as f:
        games = pickle.load(f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # each key is a data split
    # *_in splits test in-distribution generalization and *_out splits test OOD generalization
    print("Game splits:", games.keys())
    print("Data splits:", data.keys())
    print()

    # split sizes
    for split in games:
        print(f"+ {split} has {len(games[split])} games")
    for split in data:
        print(f"- {split} has {len(data[split])} datapoints")
    print()


    # keys of a datapoint
    x = data["val_in"][0]
    print(x.keys())

    # LISTENER task: x["instruction"] -> x["partial_text_obs"] + x["actions"]
    # `partial_text_obs` is a list of observation descriptions
    # it is called `partial` because it is based on a partial observation (POMDP)
    # note that the object positions are relative to the agent's pose
    print(x["partial_text_obs"][0])
    # NOTE: if there are N states, there are N - 1 actions
    print("Num states:", len(x["partial_text_obs"]), "Num actions:", len(x["actions"]))
    # to create a trajectory description, you can do the following:
    d = ""
    for o, a in zip(x["partial_text_obs"], x["actions"]):
        d += o + "\nYour action : " + a + " .\n"
    d += x["partial_text_obs"][-1]
    # this format can be helpful for prompting
    print("----------Example listener task prompt-----------")
    print("Skill description :", to_enum(Skills, x["skill_name"]).value.describe())
    print("Instruction :", x["instruction"])
    print(d)
    print("Your action:")
    print("----------End-----------")
    # for fine-tuning a transformer, consider adifferent format, perhaps without '\n'

    print()
    print()

    # SPEAKER TASK: x["partial_text_obs"] + x["actions"] -> x["skill_name"]
    # NOTE: the output is x["skill_name"], not x["instruction"]
    # x["instruction"] contains the argument of a skill (eg pick up the BLUE BALL IN ROW 6)
    # but we only want to predict the skill name

    # for prompting LLM, I also provide the skill description
    # for example
    print("----------Example speaker task prompt-----------")
    print(d)
    print()
    print("What skill does the above the trajectory describe? Below are the skill names and their descriptions")
    print()
    print("List of skills and their definitions")
    for i, s in enumerate(Skills):
        print(str(i + 1) + ". ", s.name, ":", s.value.describe())
    print("\nYour answer:")
    print("----------End prompt-----------")


def world_model_tasks():

    #game_path = "datasets/skillset_listen_games_1000.pickle"
    #data_path = "datasets/skillset_listen_data_1000.pickle"

    game_path = "datasets/temp_games.pickle"
    data_path = "datasets/temp_data.pickle"

    with open(game_path, "rb") as f:
        games = pickle.load(f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # each key is a data split
    # *_in splits test in-distribution generalization and *_out splits test OOD generalization
    print("Game splits:", games.keys())
    print("Data splits:", data.keys())
    print()

    # split sizes
    for split in games:
        print(f"+ {split} has {len(games[split])} games")
    for split in data:
        print(f"- {split} has {len(data[split])} datapoints")
    print()


    # keys of a datapoint
    x = data["val_in"][0]
    print(x.keys())

    # LISTENER task: x["init_description"] + x["edit_descriptions"] -> x["partial_text_obs"] + x["actions"]
    # x["init_description"] is the description of the initial state in the ORIGINAL environment
    # x["edit_descriptions"] is the list of edits applied to the original environment
    print(x["partial_text_obs"][0])
    # NOTE: if there are N states, there are N - 1 actions
    # this format can be helpful for prompting
    print("----------Example listener task prompt-----------")
    print("Initial environment description :", x["init_description"])
    print("The following events have sequentially happened to the environment:")
    for i, d in enumerate(x["edit_descriptions"]):
        print(f"{i + 1}.", d)
    # some trajectory prefix
    print("These are the observations and actions that the agent has seen and taken:")
    j = 2
    print("Action: start the episode")
    for o, a in zip(x["partial_text_obs"][:j], x["actions"][:j]):
        print("Observation:", o)
        print("Action:", a)
    print("Please answer the following questions (give one-word answers):")
    for i, q in enumerate(x["queries"][j]):
        print(f"{i + 1}.", q["question"])
    print("----------End-----------")

    id_to_game = {x["id"]: x for x in games["val_in"]}
    config = make_config(config_str=id_to_game[x["game_id"]]["config"])

    env = make_env(config.ai.world_model)

    # LISTENER task:  x["partial_text_obs"] + x["actions"] -> x["edit_descriptions"]
    print(x["partial_text_obs"][0])
    # NOTE: if there are N states, there are N - 1 actions
    # this format can be helpful for prompting
    print("----------Example speaker task prompt-----------")
    for o, a in zip(x["partial_text_obs"], x["actions"]):
        print("Observation:", o)
        print("Action:", a)
    print("What are the changes that occur in the environment?")
    print("Here are a list of possible changes and their definitions:")
    for i, e in enumerate(Edits):
        print(f"{i + 1}.", e.name, ":", e.value.describe(env))
    print("Your answer:")
    print("----------End-----------")




world_model_tasks()
