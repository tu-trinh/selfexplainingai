import os
import random
import yaml
import pickle
import sys
from copy import deepcopy as dc
from typing import Dict, List, Tuple, Union
from collections import Counter
from itertools import combinations
from pprint import pprint

from mindgrid.builder import make_env
from mindgrid.envs.editors import Edits
from mindgrid.infrastructure.env_utils import get_obs_desc, describe_state
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.env_constants import COLOR_NAMES
from mindgrid.planner import Planner
from mindgrid.skills import Skills
from mindgrid.infrastructure.trajectory import Trajectory


def split_attributes(
    attr: Union[List, Dict]
) -> Union[Tuple[List, List, List], Dict[str, List]]:

    def split_attr(attr: List):
        n = len(attr)
        # shuffle
        split_idx = random.sample(range(n), n)
        # split
        ret = {}
        ret["val_out"] = [attr[i] for i in split_idx[: n // 3]]
        ret["test_out"] = [attr[i] for i in split_idx[n // 3 : 2 * n // 3]]
        ret["train"] = [attr[i] for i in split_idx[2 * n // 3 :]]
        return ret

    ret = {}
    if isinstance(attr, list):
        ret = split_attr(attr)
    else:
        train_ret, val_ret, test_ret = {}, {}, {}
        for key in attr.keys():
            ret_val = split_attr(attr[key])
            for split in ret:
                ret[split][key] = ret_val[split]
    return ret


def sample_inversely(orig_counter, included):

    counter = Counter()
    for k in included:
        counter[k] = orig_counter[k]

    # Calculate the inverse of the counts
    total_sum = sum(counter.values())
    inverse_prob = {key: float(total_sum) / count for key, count in counter.items()}

    # Normalize the probabilities
    total_inverse_prob = sum(inverse_prob.values())
    probabilities = [prob / total_inverse_prob for prob in inverse_prob.values()]

    # Randomly select an element based on the computed probabilities
    return random.choices(list(counter.keys()), weights=probabilities, k=1)[0]


def execute_plan(env, plan):
    env.reset()
    t = Trajectory()
    for skill, kwargs in plan:
        t += skill.value(**kwargs)(env)
    return t


def try_generate_plan(planner, env, skillset, must_have_skill=None, n_attempts=1):
    for _ in range(n_attempts):
        plan = planner(env, skillset, must_have_skill=must_have_skill)
        if plan is not None:
            return plan
    return None


def fix_edit_order(edits):
    for i, e in enumerate(edits):
        if e == "double_grid_size":
            edits[i], edits[0] = edits[0], edits[i]
            return

def make_init_config():
    config = make_config(file_path="mindgrid/configs/base.yaml")

    # set roles
    config.roles.observer = "human"
    config.roles.executor = "human"
    config.roles.solver = "ai"

    # world model
    wm_config = config.ai.world_model
    wm_config.task = "pickup"
    wm_config.layout = random.choice(["room_door_key", "treasure_island"])
    wm_config.allow_object_colors = colors

    edits = [s.value for s in Edits]
    if wm_config.layout == "room_door_key":
        edits.remove("add_fireproof_shoes")
        edits.remove("make_lava_safe")

    skills = list(Skills)
    if wm_config.layout == "room_door_key":
        skills.remove(Skills.fix_bridge)
    elif wm_config.layout == "treasure_island":
        skills.remove(Skills.open_door)
    return config, skills



def create_skillset_datapoint(task, split, i, colors, skill_count):

    # only create dataset for listener task
    # speaker task dataset will be derived from listener task dataset
    assert task in ["listen"]

    item = {}

    config, skills = make_init_config()
    wm_config = config.ai.world_model

    must_have_skill = sample_inversely(skill_count, skills)
    print(must_have_skill.name)

    while True:

        wm_config.seed = random.randint(0, 1000000) * primes[splits.index(split)]

        if split == "train":
            n_edits = 2
        else:
            n_edits_pool = list(range(len(edits)))
            # don't select 2 edits
            n_edits_pool.remove(2)
            n_edits = random.choice(n_edits_pool)

        # NOTE: sample WITHOUT replacement
        wm_config.edits = list(random.sample(edits, n_edits))
        fix_edit_order(wm_config.edits)

        env = make_env(wm_config)
        planner = Planner(wm_config)

        p1 = try_generate_plan(planner, env, skills, must_have_skill=must_have_skill)
        if p1 is None:
            continue

        p2 = try_generate_plan(planner, env, skills)
        if p2 is None:
            continue

        p1_skillset = set([x[0] for x in p1])
        p2_skillset = set([x[0] for x in p2])

        if not p2_skillset <= p1_skillset:
            break

    assert must_have_skill in p1_skillset

    for x in p1_skillset:
        skill_count[x] += 1

    print(wm_config.edits)
    print([x.name for x in p1_skillset])
    print([x.name for x in p2_skillset])
    print(i, [(x.name, v) for x, v in list(skill_count.most_common())])
    print()

    t1 = execute_plan(env, p1)
    t2 = execute_plan(env, p2)

    config.human.world_model = config.ai.world_model = wm_config
    executor = config.roles.executor
    non_executor = "ai" if executor == "human" else "human"

    getattr(config, executor).skillset = [s.name for s in p1_skillset]
    getattr(config, non_executor).skillset = [s.name for s in p2_skillset]

    config_dict = config.to_dict()
    config_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)

    item["config"] = config_str
    item["ref_plan"] = {}
    item["ref_plan"]["executor"] = [(x[0].name, x[1]) for x in p1]
    item["ref_plan"]["non_executor"] = [(x[0].name, x[1]) for x in p2]

    return item


def create_worldmodel_datapoint(task, split, i, colors, skill_count):

    # only create dataset for listener task
    # speaker task dataset will be derived from listener task dataset
    assert task in ["listen"]

    item = {}

    config, skills = make_init_config()

    observer = config.roles.observer
    nonobserver = "human" if observer == "ai" else "ai"

    if split == "train":
        observer_edits = random.sample(edits, 2)
        nonobserver_edits = []
    else:
        nonobserver_edits = random.sample(edits, 2)
        left_edits = [x for x in edits if x not in nonobserver_edits]

        n_observer_edits = random.randint(0, len(left_edits) - 1)
        observer_edits = random.sample(left_edits, n_observer_edits)

    fix_edit_order(observer_edits)
    fix_edit_order(nonobserver_edits)

    observer_wm_config = getattr(config, observer).world_model
    nonobserver_wm_config = getattr(config, nonobserver).world_model

    observer_env = make_env(observer_wm_config)
    nonobserver_env = make_env(observer_wm_config)
    nonobserver_env.edit(nonobserver_edits)

    while True:
        n_skills = random.randint(1, len(skills) - 1)
        skillset = random.sample(skills, n_skills)

        observer_planner = Planner(

        p_observer =
        p_nonobserver =


    t1 = execute_plan(env, p1)
    t2 = execute_plan(env, p2)

    config.human.world_model = config.ai.world_model = wm_config
    executor = config.roles.executor
    non_executor = "ai" if executor == "human" else "human"

    getattr(config, executor).skillset = [s.name for s in p1_skillset]
    getattr(config, non_executor).skillset = [s.name for s in p2_skillset]

    config_dict = config.to_dict()
    config_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)

    item["config"] = config_str
    item["ref_plan"] = {}
    item["ref_plan"]["executor"] = [(x[0].name, x[1]) for x in p1]
    item["ref_plan"]["non_executor"] = [(x[0].name, x[1]) for x in p2]

    return item



def create_split(task, split, dataset, datapoint_creation_fn):

    dataset[split] = []
    n = split_size[split]
    if split == "train":
        n += split_size["val_in"] + split_size["test_in"]
    colors = color_split[split]

    skill_count = Counter()
    for s in Skills:
        skill_count[s] = 1

    for i in range(n):
        print(split.upper(), i)
        dataset[split].append(
            datapoint_creation_fn(task, split, i, colors, skill_count)
        )


def final_check(dataset):
    for s1 in dataset:
        for s2 in dataset:
            if s1 == s2:
                continue
            for item1 in dataset[s1]:
                for item2 in dataset[s2]:
                    if item1["config"] == item2["config"]:
                        print(item1["id"] + " " + item2["id"])
                        print(item1["config"])
                        print(item2["config"])
                        sys.exit(1)


def print_stats(dataset):
    for s in dataset:
        print(f" * Split {s}: {len(dataset[s])} datapoints")


def create_dataset(prefix, task, save_path, datapoint_creation_fn):

    dataset = {}
    for split in splits:
        if "_in" in split:
            continue
        create_split(task, split, dataset, datapoint_creation_fn)

    dataset["test_in"] = dataset["train"][-split_size["test_in"] :]
    dataset["val_in"] = dataset["train"][
        -(split_size["val_in"] + split_size["test_in"]) : -split_size["test_in"]
    ]
    dataset["train"] = dataset["train"][
        : -(split_size["val_in"] + split_size["test_in"])
    ]

    # assign id
    for split in dataset:
        for i, item in enumerate(dataset[split]):
            item["id"] = f"{prefix}-{task}-{split}-{i}"

    final_check(dataset)
    print_stats(dataset)

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved data to {save_path}!")

    with open(save_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Reload dataset successful!")


if __name__ == "__main__":

    random.seed(100)
    color_split = split_attributes(COLOR_NAMES)

    splits = ["train", "val_in", "test_in", "val_out", "test_out"]
    split_size = {
        "train": 1000,
        "val_in": 100,
        "test_in": 100,
        "val_out": 100,
        "test_out": 100,
    }
    primes = [2, 3, 5, 7, 11]

    prefix = "skillset"
    task = "listen"
    #save_path = "datasets/temp.pickle"
    save_path = f"datasets/{prefix}_{task}_games_{split_size['train']}.pickle"

    if os.path.exists(save_path):
        print(f"File {save_path} exists!")
        sys.exit(1)

    create_dataset(prefix, task, save_path, create_skillset_datapoint)
