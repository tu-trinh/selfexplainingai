from package.enums import Task, Level
from package.builder import make_agents
from package.infrastructure.basic_utils import debug, format_seconds

import yaml
import time
import pickle
import copy


"""
Belief Mismatch Dataset
"""


"""
Intention Mismatch Dataset

Train   Test
------------
A1 A2  A3 A4 \
B1 B2  B3 B4 / same layout, diff skills
------------
C1 C2  D1 D2 \
C3 C4  D3 D4 / diff layout, same skills
------------
E1 E2  F5 F6 \
E3 E4  F7 F8 / diff layout, diff skills

For odd skillsets: A has narrower skillset than P (higher alpha)
skillset   alpha_P   alpha_A
----------------------------
    1        0.0       1.0
    3        0.3       0.5
    5        0.5       0.7
    7        0.7       0.9
For even skillsets: A has broader skillset than P (lower alpha)
skillset   alpha_P   alpha_A
----------------------------
    2        0.5       0.3
    4        0.7       0.5
    6        0.9       0.7
    8        1.0       0.0
"""
INTENTION_COLUMNS = ["config", "mission", "skill", "setup_actions", "trajectory"]

task_to_pref_mapping = {
    Task.GOTO: "reward_reach_object_hof",
    Task.PICKUP: "reward_carry_object_hof",
    Task.PUT: "reward_adjacent_object_hof",
    Task.COLLECT: "reward_adjacent_object_hof",
    Task.CLUSTER: "reward_adjacent_object_hof"
}
seed_alpha_sets = {  # holds each block of four. see reference above. for example, A1 corresponds to seed A = 1 and skillset 1 (aka alpha_p = 0 and alpha_a = 1)
    "train": [
        (1, 0, 1), (1, 0.5, 0.3), (2, 0, 1), (2, 0.5, 0.3),
        (3, 0, 1), (3, 0.5, 0.3), (3, 0.3, 0.5), (3, 0.7, 0.5),
        (5, 0, 1), (5, 0.5, 0.3), (5, 0.3, 0.5), (5, 0.7, 0.5)
    ],
    "test": [
        (1, 0.3, 0.5), (1, 0.7, 0.5), (2, 0.3, 0.5), (2, 0.7, 0.5),
        (4, 0, 1), (4, 0.5, 0.3), (4, 0.3, 0.5), (4, 0.7, 0.5),
        (6, 0.5, 0.7), (6, 0.9, 0.7), (6, 0.7, 0.9), (6, 1, 0)
    ]
}

debug_render = False
greenlight = lambda t, l, k, s, ap, aa: l not in [Level.BOSS] and t in [Task.GOTO, Task.PICKUP]
# greenlight = lambda t, l, k, s, ap, aa: (t == Task.GOTO) and (l == Level.TREASURE_ISLAND) and (s == 1) and (ap == 0) and (aa == 1)
testing = False

def create_datasets(mismatch):  # loose upper bound: 5 tasks x 12 levels x 2 train/test x 12 sets x 2 secs = 48 min
    start = time.time()
    for task in [Task.GOTO, Task.PICKUP]:
        for level in Level:
            for key in seed_alpha_sets:
                temp_dataset = {col: [] for col in INTENTION_COLUMNS}
                for seed, alpha_p, alpha_a in seed_alpha_sets[key]:
                    if greenlight(task, level, key, seed, alpha_p, alpha_a):
                        print(f"Starting {task.value} - {level.value} - {key} - {(seed, alpha_p, alpha_a)}")
                        env_dict = {
                            "task": task.value.upper(),
                            "principal_level": level.value.upper(),
                            "seed": seed
                        }
                        p_dict = {
                            "query_source": "openai",
                            "basic_reward_functions": [{"name": task_to_pref_mapping[task]}],
                            "basic_reward_weights": [1],
                            "skills": [""]
                        }
                        a_dict = {
                            "query_source": "openai",
                            "model_source": "gpt",
                            "basic_reward_functions": [{"name": task_to_pref_mapping[task]}],
                            "basic_reward_weights": [1],
                            "skills": [""]
                        }
                        yaml_obj = {"principal": p_dict, "attendant": a_dict, "env_specs": env_dict}
                        p, a = make_agents(config = copy.deepcopy(yaml_obj))
                        if debug_render:
                            p.world_model.render_mode = "human"
                            p.world_model.render()
                            time.sleep(100)
                        p._find_optimal_policy()
                        p._build_task_tree()
                        # p_skills = p.task_tree.get_skill_subset(alpha_p)
                        a_skills = p.task_tree.get_skill_subset(alpha_a)
                        # yaml_obj["principal"]["skills"] = p_skills
                        yaml_obj["attendant"]["skills"] = a_skills
                        yaml_str = yaml.dump(yaml_obj)
                        for skill in a_skills:
                            setup_actions, actions = a._retrieve_actions_from_skill_func(skill)
                            temp_dataset["config"].append(yaml_str)
                            temp_dataset["mission"].append(p.world_model.mission)
                            temp_dataset["skill"].append(skill)
                            temp_dataset["setup_actions"].append(setup_actions)
                            temp_dataset["trajectory"].append(actions)
                        print("Finished")
                        if testing:
                            return
                if len(temp_dataset["mission"]) > 0:
                    with open(f"./datasets/{mismatch}/{task.value}_{level.value}_{key}.pkl", "wb") as f:
                        pickle.dump(temp_dataset, f)
    end = time.time()
    print(format_seconds(end - start))

def join_datasets(mismatch):
    mega_train = {col: [] for col in INTENTION_COLUMNS}
    mega_test = {col: [] for col in INTENTION_COLUMNS}
    dataset_dir = f"./datasets/{mismatch}/"
    for task in Task:
        for level in Level:
            for key in seed_alpha_sets:
                if greenlight(task, level, key, 0, 0, 0):
                    ds_path = dataset_dir + f"{task.value}_{level.value}_{key}.pkl"
                    with open(ds_path, "rb") as f:
                        ds_df = pickle.load(f)
                        if key == "train":
                            add_to = mega_train
                        else:
                            add_to = mega_test
                        for col in INTENTION_COLUMNS:
                            add_to[col].extend(ds_df[col])
    dataset = {"train": mega_train, "test": mega_test}
    with open(dataset_dir + f"{mismatch}_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    

# create_datasets("intention")
join_datasets("intention")


"""
Reward Mismatch Dataset
"""