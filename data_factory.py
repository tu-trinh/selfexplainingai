import os
import sys
import pickle
import random
import inflect
from nltk.tokenize import word_tokenize

from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import to_enum
from mindgrid.infrastructure.env_utils import describe_state, describe_object_state, describe_object, get_attribute
from mindgrid.infrastructure.env_constants import IDX_TO_DIR
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.skills import Skills


def tokenize(s):
    return " ".join(word_tokenize(s))

def create_skillset_datapoint(game_item):

    config = make_config(config_str=game_item["config"])

    env = make_env(getattr(config, config.roles.executor).world_model)
    env.reset()

    plan = game_item["ref_plan"][config.roles.executor]

    data_points = []

    for s, a in plan:

        item = {}
        item["game_id"] = game_item["id"]
        item["skill_name"] = s
        item["arguments"] = a

        skill_cls = to_enum(Skills, s).value
        skill = skill_cls(**a)
        item["instruction"] = tokenize(tokenize(skill.verbalize(env)))

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


def create_worldmodel_datapoint(game_item):

    config = make_config(config_str=game_item["config"])
    observer = config.roles.observer
    nonobserver = "human" if observer == "ai" else "ai"

    item = {}
    item["game_id"] = game_item["id"]

    observer_env = make_env(getattr(config, observer).world_model)
    nonobserver_env = make_env(getattr(config, nonobserver).world_model)

    nonobserver_env.reset()
    observer_env.reset()

    print(getattr(config, observer).world_model.edits)
    """
    observer_env.render_mode = "human"
    observer_env.render()
    """
    #input()

    item["init_description"] = describe_state(nonobserver_env.get_state(), relative=False)
    item["edit_descriptions"] = []
    for e in observer_env.applied_edits:
        item["edit_descriptions"].append(tokenize(e.verbalize()))

    plan = game_item["ref_plan"][config.roles.observer]
    t = Trajectory()
    print(plan)
    for s, a in plan:
        t += to_enum(Skills, s).value(**a)(observer_env)


    print(t.n_states)
    #input()

    item["actions"] = [x.name for x in t.actions]
    item["states"] = t.states

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

    item["full_obs"] = full_obs
    item["full_text_obs"] = full_text_obs
    item["partial_obs"] = partial_obs
    item["partial_text_obs"] = partial_text_obs

    print(t.n_actions)

    item["queries"] = []
    for j in range(t.n_actions + 1):
        item["queries"].append([])
        for i in range(5):
            a = -1
            while a == -1:
                state = t.states[j]

                types = set([o.type for o in state.objects])
                o_type = random.choice(list(types))
                objects_type = [o for o in state.objects if o.type == o_type] + ["agent"]

                o = random.choice(objects_type)

                if o == "agent":
                    cand_attrs = ["dir", "x", "y"]
                    attr = random.choice(cand_attrs)
                    if attr == "x":
                        attr_name = "column"
                        a = state.agent_pos[0]
                    elif attr == "y":
                        attr_name = "row"
                        a = state.agent_pos[1]
                    elif attr == "dir":
                        attr_name = "facing direction"
                        a = IDX_TO_DIR[state.agent_dir]

                    q = f"what is the {attr_name} of the agent?"

                else:
                    cand_attrs = ["x", "y", "color", "state"]
                    if describe_object_state(o) == "":
                        cand_attrs.remove("state")
                    attr = random.choice(cand_attrs)

                    attr_name = attr
                    if attr == "x":
                        attr_name = "column"
                    elif attr_name == "y":
                        attr_name = "row"

                    cand_attrs.remove(attr)
                    q = f"what is the {attr_name} of the {describe_object(o, state.objects, relative=False, specified_attrs=cand_attrs)}?"

                    a = get_attribute(o, attr)

            print(q, a)
            item["queries"][-1].append({ "time_step": j, "object": o, "attribute": attr, "question": q, "answer": a})

    return [item]

def create_dataset(prefix, task, game_dataset, datapoint_creation_fn):
    dataset = {}

    for split in game_dataset:
        dataset[split] = []
        for i, item in enumerate(game_dataset[split]):
            print(split.upper(), i)
            dataset[split].extend(datapoint_creation_fn(item))

        for i, item in enumerate(dataset[split]):
            item["id"] = f"{prefix}_{task}_{split}_i"

    for split in dataset:
        print(f" * Split {split}: {len(dataset[split])} datapoints")

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved data to {save_path}!")

    with open(save_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Reload dataset successful!")


random.seed(340)

version = 1
#prefix = "skillset"
prefix = "worldmodel"
task = "listen"

#open_path = f"datasets/{prefix}_{task}_games_1000_v{version}.pickle"
open_path = f"datasets/temp_games.pickle"

save_path = open_path.replace("games", "data")

if os.path.exists(save_path):
    print(f"File {save_path} exists!")
    sys.exit(1)

with open(open_path, "rb") as f:
    game_dataset = pickle.load(f)

if prefix == "skillset":
    create_dataset(prefix, task, game_dataset, create_skillset_datapoint)
elif prefix == "worldmodel":
    create_dataset(prefix, task, game_dataset, create_worldmodel_datapoint)


