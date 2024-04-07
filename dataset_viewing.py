from package.enums import Task, Level
from package.builder import make_agents
from package.infrastructure.basic_utils import debug

from environment_play import manual_test

import pickle
import pandas as pd
import yaml
import time

dspath = "./datasets/intention/Goto_Empty_train.pkl"
with open(dspath, "rb") as f:
    ds = pickle.load(f)

print(len(ds["config"]), "total datapoints")
i = 4
dataset_test_config = ds["config"][i]
with open("./dataset_test_config.yaml", "w") as f:
    f.write(dataset_test_config)
p, a = make_agents(config_path = "./dataset_test_config.yaml")
print("SKILL:", ds["skill"][i])
print("SETUP ACTIONS:", ds["setup_actions"][i])
print("TRAJECTORY:", ds["trajectory"][i])
env = p.world_model
env.render_mode = "human"
env.render()
time.sleep(100)