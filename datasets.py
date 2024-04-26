from package.enums import Task, Level
from package.builder import make_agents
from package.infrastructure.basic_utils import debug, format_seconds
from package.infrastructure.env_constants import (
    COLOR_NAMES,
    MIN_ROOM_SIZE,
    MAX_ROOM_SIZE,
    SKILL_PHRASES,
)

import yaml
import time
import pickle
import copy
import random
from typing import Union, List, Dict, Tuple


train_seeds = [1]  # [1, 2, 3, 5, 7, 9]
val_seeds = [1]  # [1, 2, 11, 12, 13, 14]
test_seeds = [1]  # [1, 2, 4, 6, 8, 10]
task_to_pref_mapping = {
    Task.GOTO: "reward_reach_object_hof",
    Task.PICKUP: "reward_carry_object_hof",
    Task.PUT: "reward_adjacent_object_hof",
    Task.COLLECT: "reward_adjacent_object_hof",
    Task.CLUSTER: "reward_adjacent_object_hof",
}


"""
Belief Mismatch Dataset
"""


"""
Intention Mismatch Dataset
"""
INTENTION_COLUMNS = [
    "config",
    "mission",
    "skill",
    "actions",
    "traj_fully_obs_grid",
    "traj_fully_obs_text",
    "traj_partial_obs_grid",
    "traj_partial_obs_text",
]


def split_attributes(
    attr: Union[List, Dict]
) -> Union[Tuple[List, List, List], Dict[str, List]]:
    def split_attr(attr: List):
        n = len(attr)
        split_idx = random.sample(range(n), n)
        val_values = [attr[i] for i in split_idx[: n // 3]]
        test_values = [attr[i] for i in split_idx[n // 3 : 2 * n // 3]]
        train_values = [attr[i] for i in split_idx[2 * n // 3 :]]
        return train_values, val_values, test_values

    if isinstance(attr, list):
        return split_attr(attr)
    else:
        train_ret, val_ret, test_ret = {}, {}, {}
        for key in attr.keys():
            train, val, test = split_attr(attr[key])
            train_ret[key] = train
            val_ret[key] = val
            test_ret[key] = test
        return train_ret, val_ret, test_ret


def make_config(task: Task, level: Level, seed: int, colors: List[str]) -> Dict:
    env_dict = {
        "task": task.value.upper(),
        "principal_level": level.value.upper(),
        "seed": seed,
        "allowed_object_colors": colors,
    }
    p_dict = {
        "basic_reward_functions": [{"name": task_to_pref_mapping[task]}],
        "basic_reward_weights": [1],
        "skills": [""],
    }
    a_dict = {
        "basic_reward_functions": [{"name": task_to_pref_mapping[task]}],
        "basic_reward_weights": [1],
        "skills": [""],
    }
    yaml_obj = {"principal": p_dict, "attendant": a_dict, "env_specs": env_dict}
    return yaml_obj


def filter_skills(skills: List[str], steps: List[int]) -> List[str]:
    filtered_skills = []
    for skill in skills:
        if "move_" in skill:
            if int(skill.split("_")[2]) in steps:
                filtered_skills.append(skill)
        else:
            filtered_skills.append(skill)
    return filtered_skills


def get_skill_phrase_key(skill: str) -> str:
    skill_split = skill.split("_")
    if len(skill_split) == 1:  # primitive skill
        return skill
    if "move" in skill:
        return "move_" + skill_split[1]
    return skill_split[0] + "_"


def save_dataset(mismatch: str, task: Task, level: Level, ds: str, dataset: Dict):
    ds_name = f"{task.value}_{level.value}_{ds}"
    if len(dataset["mission"]) > 0:
        with open(f"./datasets/{mismatch}/{ds_name}.pkl", "wb") as f:
            pickle.dump(dataset, f)
    debug("Saved", ds_name, "\n")


def create_datasets(
    mismatch,
):  # loose upper bound: 5 tasks x 12 levels x 2 train/test x 12 sets x 2 secs = 48 min
    start = time.time()
    for task in [Task.PICKUP]:
        for level in [Level.ROOM_DOOR_KEY, Level.TREASURE_ISLAND]:
            # Split up attributes for train/val/test
            train_colors, val_colors, test_colors = split_attributes(COLOR_NAMES)
            train_steps, val_steps, test_steps = split_attributes(
                list(range(MIN_ROOM_SIZE - 2, MAX_ROOM_SIZE - 1))
            )
            train_phrases, val_phrases, test_phrases = split_attributes(SKILL_PHRASES)

            # Make datasets
            train_dataset = add_to_dataset(
                task,
                level,
                train_colors,
                train_steps,
                train_phrases,
                train_seeds,
                "train",
            )
            if not testing:
                save_dataset(mismatch, task, level, "train", train_dataset)

            """
            val_dataset = add_to_dataset(task, level, val_colors, val_steps, val_phrases, val_seeds, "val", ref_colors = train_colors, ref_steps = train_steps, ref_phrases = train_phrases)
            if not testing:
                save_dataset(mismatch, task, level, "val", val_dataset)

            test_dataset = add_to_dataset(task, level, test_colors, test_steps, test_phrases, test_seeds, "test", ref_colors = train_colors, ref_steps = train_steps, ref_phrases = train_phrases)
            if not testing:
                save_dataset(mismatch, task, level, "test", test_dataset)
            """
    end = time.time()
    print(format_seconds(end - start))


debug_render = False
greenlight = lambda t, l, s: True
# greenlight = lambda t, l, s: (t == Task.GOTO) and (l == Level.TREASURE_ISLAND) and (s == 1) and (ap == 0) and (aa == 1)
testing = True


def add_to_dataset(
    task: Task,
    level: Level,
    colors: List[str],
    steps: List[int],
    phrases: Dict,
    seeds: List[int],
    ds: str,
    ref_colors: List[str] = None,
    ref_steps: List[int] = None,
    ref_phrases: List[str] = None,
) -> Dict:
    dataset = {col: [] for col in INTENTION_COLUMNS}
    for i, seed in enumerate(seeds):
        if greenlight(task, level, seed):
            print(f"[{ds.upper()}] Starting {task.value} - {level.value} - seed {seed}")

            if ds == "train" or i not in [2, 3]:
                yaml_obj = make_config(task, level, seed, colors)
            else:
                yaml_obj = make_config(task, level, seed, ref_colors)
            _, a = make_agents(config=copy.deepcopy(yaml_obj))

            if debug_render:
                ogrm = a.world_model.render_mode
                a.world_model.render_mode = "human"
                a.world_model.render()
                time.sleep(100)
                a.world_model.render_mode = ogrm

            a._find_optimal_policy()
            a._build_task_tree()
            a_skills = a.task_tree.get_skill_subset(
                seed / 10 if seed <= 10 else seed / 20
            )
            if ds == "train" or i not in [2, 3]:
                filtered_a_skills = filter_skills(a_skills, steps)
            else:
                filtered_a_skills = filter_skills(a_skills, ref_steps)
            yaml_obj["attendant"]["skills"] = filtered_a_skills

            for skill in filtered_a_skills:
                debug(f"DEMONSTRATE {skill}")
                setup_actions, actions = a._retrieve_actions_from_skill_func(skill)
                debug(actions)
                yaml_obj["env_specs"]["attendant_setup_actions"] = setup_actions
                yaml_str = yaml.dump(yaml_obj)
                mission = a.world_model.mission
                traj_fully_obs_grid = a._generate_obs_act_sequence(
                    actions, setup_actions, as_text=False, fully_obs=True
                )
                traj_fully_obs_text = a._generate_obs_act_sequence(
                    actions, setup_actions, as_text=True, fully_obs=True
                )
                traj_partial_obs_grid = a._generate_obs_act_sequence(
                    actions, setup_actions, as_text=False, fully_obs=False
                )
                traj_partial_obs_text = a._generate_obs_act_sequence(
                    actions, setup_actions, as_text=True, fully_obs=False
                )
                if ds == "train" or i not in [2, 3]:
                    paraphrases = phrases
                else:
                    paraphrases = ref_phrases
                for paraphrase in paraphrases[get_skill_phrase_key(skill)]:
                    dataset["config"].append(yaml_str)
                    dataset["mission"].append(mission)
                    dataset["skill"].append(paraphrase(skill))
                    dataset["actions"].append(actions)
                    dataset["traj_fully_obs_grid"].append(traj_fully_obs_grid)
                    dataset["traj_fully_obs_text"].append(traj_fully_obs_text)
                    dataset["traj_partial_obs_grid"].append(traj_partial_obs_grid)
                    dataset["traj_partial_obs_text"].append(traj_partial_obs_text)
    return dataset


def join_datasets(mismatch):
    megasets = {
        "train": {col: [] for col in INTENTION_COLUMNS},
        "val": {col: [] for col in INTENTION_COLUMNS},
        "test": {col: [] for col in INTENTION_COLUMNS},
    }
    dataset_dir = f"./datasets/{mismatch}/"
    for task in Task:
        for level in Level:
            for key in megasets:
                ds_path = dataset_dir + f"{task.value}_{level.value}_{key}.pkl"
                try:
                    with open(ds_path, "rb") as f:
                        ds_df = pickle.load(f)
                        add_to = megasets[key]
                        for col in INTENTION_COLUMNS:
                            add_to[col].extend(ds_df[col])
                except FileNotFoundError:
                    pass
    """
    with open(f"./datasets/{mismatch}_datasets.pkl", "wb") as f:
        pickle.dump(megasets, f)
    """


create_datasets("intention")
# join_datasets("intention")


"""
Reward Mismatch Dataset
"""
