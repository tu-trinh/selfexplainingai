import os
import sys
import pickle
from nltk.tokenize import word_tokenize

from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.infrastructure.env_utils import describe_state
from mindgrid.skills import Skills


def tokenize(s):
    return " ".join(word_tokenize(s))

def create_intention_datapoint(game_item):

    config = make_config(config_str=game_item["config"])
    env = make_env(getattr(config, config.roles.executor).world_model)
    env.reset()

    plan = game_item["ref_plan"]["executor"]

    data_points = []

    for s, a in plan:

        item = {}
        item["game_id"] = game_item["id"]
        item["skill_name"] = s
        item["arguments"] = a

        skill = to_enum(Skills, s).value(**a)
        item["instruction"] = tokenize(skill.verbalize(env))
        item["skill_description"] = tokenize(skill.description())

        t = skill(env)
        full_obs = []
        full_text_obs = []
        partial_obs = []
        partial_text_obs = []
        for i in range(t.n_states):
            state = t.states[i]

            full_obs.append(state.full_obs)
            full_text_obs.append(tokenize(describe_state(state, relative=False)))

            partial_obs.append(state.partial_obs)
            partial_text_obs.append(tokenize(describe_state(state, relative=True)))

        item["states"] = t.states
        item["full_obs"] = full_obs
        item["full_text_obs"] = full_text_obs
        item["partial_obs"] = partial_obs
        item["partial_text_obs"] = partial_text_obs
        item["actions"] = [x.name for x in t.actions]


        #print(item["instruction"])
        #print(item["skill_description"])
        #print(item["full_text_obs"][0])
        #print(item["partial_text_obs"][0])

        data_points.append(item)

    assert (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )

    return data_points


open_path = "datasets/skillset_listen_games_1000.pickle"
save_path = open_path.replace("games", "data")

if os.path.exists(save_path):
    print(f"File {save_path} exists!")
    sys.exit(1)

with open(open_path, "rb") as f:
    game_dataset = pickle.load(f)

dataset = {}

for split in game_dataset:
    dataset[split] = []
    for i, item in enumerate(game_dataset[split]):
        print(split.upper(), i)
        dataset[split].extend(create_intention_datapoint(item))

for split in dataset:
    print(f" * Split {split}: {len(dataset[split])} datapoints")

with open(save_path, "wb") as f:
    pickle.dump(dataset, f)
print(f"Saved data to {save_path}!")

with open(save_path, "rb") as f:
    dataset = pickle.load(f)
print(f"Reload dataset successful!")

