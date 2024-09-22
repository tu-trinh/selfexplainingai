import os
import argparse
import pickle
import random
import sys
from collections import Counter
from copy import deepcopy as dc
from itertools import combinations
from pprint import pprint
from typing import Dict, List, Tuple, Union

import yaml

from mindgrid.builder import make_env
from mindgrid.envs.edits import Edits
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.env_constants import COLOR_NAMES
from mindgrid.infrastructure.env_utils import describe_state, get_obs_desc
from mindgrid.infrastructure.trajectory import Trajectory
from mindgrid.planner import Planner
from mindgrid.skills import Skills


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
    assert (
        env.carrying is not None
        and env.carrying.type == "ball"
        and env.carrying.color == env.target_color
    )
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


def make_init_config(colors):
    config = make_config(file_path="mindgrid/configs/base.yaml")

    # env
    env_config = config.false_agent.env
    env_config.task = "pickup"
    env_config.layout = random.choice(["room_door_key", "treasure_island"])
    env_config.allowed_object_colors = colors

    edits = [s.name for s in Edits]
    if env_config.layout == "room_door_key":
        edits.remove("add_fireproof_shoes")
        edits.remove("make_lava_safe")

    skills = list(Skills)
    if env_config.layout == "room_door_key":
        skills.remove(Skills.fix_bridge)
    elif env_config.layout == "treasure_island":
        skills.remove(Skills.open_door)

    return config, edits, skills


def create_skill_datapoint(task, split, i, colors, skill_count):

    print(colors)

    # only create dataset for listener task
    # speaker task dataset will be derived from listener task dataset
    assert task in ["listen"]

    item = {}

    config, edits, skills = make_init_config(colors)
    env_config = config.ai.world_model

    must_have_skill = sample_inversely(skill_count, skills)
    print(must_have_skill.name)

    while True:

        env_config.seed = random.randint(0, 1000000) * primes[splits.index(split)]

        if split == "train":
            n_edits = 2
        else:
            n_edits_pool = list(range(len(edits)))
            # don't select 2 edits
            n_edits_pool.remove(2)
            n_edits = random.choice(n_edits_pool)

        # NOTE: sample WITHOUT replacement
        env_config.edits = list(random.sample(edits, n_edits))
        fix_edit_order(env_config.edits)

        env = make_env(env_config)
        planner = Planner(env_config)

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

    print(env_config.edits)
    print([x.name for x in p1_skillset])
    print([x.name for x in p2_skillset])
    print(i, [(x.name, v) for x, v in list(skill_count.most_common())])
    print()

    t1 = execute_plan(env, p1)
    t2 = execute_plan(env, p2)

    config.human.world_model = config.ai.world_model = env_config
    executor = config.roles.executor
    nonexecutor = "ai" if executor == "human" else "human"

    getattr(config, executor).skillset = [s.name for s in p1_skillset]
    getattr(config, nonexecutor).skillset = [s.name for s in p2_skillset]

    config_dict = config.to_dict()
    config_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)

    item["config"] = config_str
    item["ref_plan"] = {}
    item["ref_plan"][executor] = [(x[0].name, x[1]) for x in p1]
    item["ref_plan"][nonexecutor] = [(x[0].name, x[1]) for x in p2]


    return item


def create_env_datapoint(split, i, colors, skill_count):

    item = {}

    config, edits, skills = make_init_config(colors)

    while True:

        config.false_agent.env.seed = random.randint(0, 1000000) * primes[splits.index(split)]
        config.true_agent.env = config.false_agent.env.clone()

        if split == "train":
            false_agent_edits = []
            true_agent_edits = random.sample(edits, 2)
            fix_edit_order(true_agent_edits)
        else:
            false_agent_edits = random.sample(edits, 2)
            fix_edit_order(false_agent_edits)
            left_edits = [x for x in edits if x not in false_agent_edits and x != "double_grid_size"]
            n_true_agent_edits = random.randint(0, len(left_edits) - 1)
            true_agent_edits = false_agent_edits + random.sample(left_edits, n_true_agent_edits)

        config.true_agent.env.edits = true_agent_edits
        config.false_agent.env.edits = false_agent_edits

        true_agent_env_config = config.true_agent.env
        true_agent_env = make_env(true_agent_env_config)
        true_agent_planner = Planner(true_agent_env_config)
        p_true_agent = true_agent_planner(true_agent_env, skills)

        false_agent_env_config = config.false_agent.env
        false_agent_env = make_env(false_agent_env_config)
        false_agent_planner = Planner(false_agent_env_config)
        p_false_agent = false_agent_planner(false_agent_env, skills)

        if p_true_agent is not None and p_false_agent is not None:
            break

    print(true_agent_edits)
    print(false_agent_edits)

    # NOTE: re-create environment and make sure plan works!
    true_agent_env = make_env(config.true_agent.env)
    execute_plan(true_agent_env, p_true_agent)
    false_agent_env = make_env(config.false_agent.env)
    execute_plan(false_agent_env, p_false_agent)

    config.true_agent.skill = [s.name for s in skills]
    config.false_agent.skill = [s.name for s in skills]

    config_dict = config.to_dict()
    config_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)
    item["config"] = config_str
    item["ref_plan"] = {}
    item["ref_plan"]["true_agent"] = [(x[0].name, x[1]) for x in p_true_agent]
    item["ref_plan"]["false_agent"] = [(x[0].name, x[1]) for x in p_false_agent]

    return item


def create_split(split, dataset, datapoint_creation_fn):

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
            datapoint_creation_fn(split, i, colors, skill_count)
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


def create_dataset(prefix, save_path, datapoint_creation_fn):

    dataset = {}
    for split in splits:
        if "_in" in split:
            continue
        create_split(split, dataset, datapoint_creation_fn)

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
            item["id"] = f"{prefix}-{split}-{i}"

    final_check(dataset)
    print_stats(dataset)

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved data to {save_path}!")

    with open(save_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Reload dataset successful!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    random.seed(100)
    color_split = split_attributes(COLOR_NAMES)

    splits = ["train", "val_in", "test_in", "val_out", "test_out"]

    primes = [2, 3, 5, 7, 11]

    version = args.version
    #prefix = "skill"
    prefix = args.prefix

    if prefix == "skill":
        split_size = {
            "train": 1000,
            "val_in": 100,
            "test_in": 100,
            "val_out": 100,
            "test_out": 100,
        }

    elif prefix == "env":
        split_size = {
            "train": 5000,
            "val_in": 500,
            "test_in": 500,
            "val_out": 500,
            "test_out": 500,
        }

    #save_path = "datasets/temp_games.pickle"
    save_path = f"datasets/{prefix}_games_{split_size['train']}_v{version}.pickle"

    print(f"Save to {save_path}?")
    input()

    if os.path.exists(save_path):
        print(f"File {save_path} exists!")
        sys.exit(1)
    else:
        print(f"Will save to {save_path}")

    if prefix == "skill":
        create_dataset(prefix, save_path, create_skill_datapoint)
    elif prefix == "env":
        create_dataset(prefix, save_path, create_env_datapoint)
