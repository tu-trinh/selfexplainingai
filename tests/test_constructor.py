from copy import deepcopy as dc
import yaml
import numpy as np
import sys
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

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
                if task == "PICKUP":
                    config["principal"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                    config["attendant"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                elif task in ["PUT", "COLLECT", "CLUSTER"]:
                    config["principal"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                    config["attendant"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                try:
                    make_agents(config=config)
                    results.append(0)
                except (AssertionError, ValueError) as e:
                    print(f"ASSERT: {task}, principal = {p_level}, attendant = {a_level}")
                    print(e)
                    results.append(1)
                except Exception as e:
                    print(f"FAILED: {task}, principal = {p_level}, attendant = {a_level}")
                    print(e)
                    results.append(2)

    results = np.array(results)
    print("Failure rate:", np.count_nonzero(results == 2) / len(results))
    print("AssertionError rate:", np.count_nonzero(results == 1) / len(results))

    assert sum(results) == 0
