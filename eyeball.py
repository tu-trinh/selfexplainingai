import pickle

from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from minigrid.core.actions import Actions


def eyeball_skillset():
    game_path = "datasets/skillset_listen_games_1000_v2.pickle"
    data_path = "datasets/skillset_listen_data_1000_v2.pickle"

    with open(game_path, "rb") as f:
        games = pickle.load(f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # NOTE: change split
    split = "train"

    id_to_game = {x["id"] : x for x in games[split]}

    # NOTE: change the id of the example
    x = data[split][234]

    game = id_to_game[x["game_id"]]

    config = make_config(config_str=game["config"])

    env = make_env(getattr(config, config.roles.executor).world_model)

    env.render_mode = "human"

    env.reset_from_state(x['states'][0])

    env.render()

    for s, a in zip(x["partial_text_obs"], x["actions"]):
        print("Instruction:", x["instruction"])
        print("Current observation:", s)
        print("Next action:", a)
        print()
        action = getattr(Actions, a)

        input()
        env.step(action)
        env.render()
    print("Current observation:", x["partial_text_obs"][-1])
    input()


def eyeball_worldmodel():
    game_path = "datasets/worldmodel_listen_games_5000_v2.pickle"
    data_path = "datasets/worldmodel_listen_data_5000_v2.pickle"

    with open(game_path, "rb") as f:
        games = pickle.load(f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # NOTE: change split
    split = "train"

    id_to_game = {x["id"] : x for x in games[split]}

    # NOTE: change the id of the example
    x = data[split][2]

    game = id_to_game[x["game_id"]]

    config = make_config(config_str=game["config"])

    env = make_env(getattr(config, config.roles.executor).world_model)

    env.render_mode = "human"

    env.reset_from_state(x['states'][0])

    env.render()

    print("Edits:")
    for e in x["edit_descriptions"]:
        print(e)
    print()

    for s, a in zip(x["partial_text_obs"], x["actions"]):
        print("Current observation:", s)
        print("Next action:", a)
        print()
        action = getattr(Actions, a)

        input()
        env.step(action)
        env.render()
    print("Current observation:", x["partial_text_obs"][-1])
    input()


# NOTE: uncomment one of the followings
eyeball_skillset()
# eyeball_worldmodel()
