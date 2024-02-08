from copy import deepcopy as dc
import yaml
import numpy
import sys

sys.path.append(".")

from package.enums import *
from package.builder import *


def test_construct_all_env_types_and_levels():
    config_file = "package/configs/test1_difficulty1.yaml"
    with open(config_file, "r") as f:
        orig_config = yaml.load(f, Loader = yaml.SafeLoader)

    results = []
    for x in EnvType:
        task = str(x).replace("EnvType.", "")
        for y in Level:
            p_level = str(y).replace("Level.", "")
            for z in Level:
                a_level = str(z).replace("Level.", "")
                config = dc(orig_config)
                config["env_specs"]["task"] = task
                config["env_specs"]["principal_level"] = p_level
                config["env_specs"]["attendant_level"] = a_level
                print(task, p_level, a_level)
                try:
                    make_agents(config=config)
                    results.append(0)
                except:
                    results.append(1)
                print("Failure rate:", sum(results), "/", len(results))

    assert sum(results) == 0
