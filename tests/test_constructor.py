from copy import deepcopy as dc
import yaml
import numpy as np
import sys
import traceback
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.enums import *
from package.builder import *


results = []

def test_construct_all_tasks_and_levels(test_levels):
    global results

    config_file = "package/configs/intention_speaker_difficulty1.yaml"
    with open(config_file, "r") as f:
        orig_config = yaml.load(f, Loader = yaml.SafeLoader)

    for x in Task:
        task = str(x).replace("Task.", "")
        if test_levels:
            for y in Level:
                p_level = str(y).replace("Level.", "")
                for z in Level:
                    a_level = str(z).replace("Level.", "")
                    config = dc(orig_config)
                    config["env_specs"]["task"] = task
                    config["env_specs"]["principal_level"] = p_level
                    config["env_specs"]["attendant_level"] = a_level
                    if task == "GOTO":
                        config["principal"]["basic_reward_functions"] = [{"name": "reward_reach_object_hof"}]
                        config["attendant"]["basic_reward_functions"] = [{"name": "reward_reach_object_hof"}]
                    elif task == "PICKUP":
                        config["principal"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                        config["attendant"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                    elif task in ["PUT", "COLLECT", "CLUSTER"]:
                        config["principal"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                        config["attendant"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                    del config["env_specs"]["attendant_edits"]
                    if not test_control(config, task, p_level = p_level, a_level = a_level):
                        return
        else:
            with open("package/configs/edits.txt", "r") as f:
                possible_edits = [pe.strip() for pe in f.readlines()]
            curr_group = "General"
            for pe in possible_edits:
                if pe.endswith(":"):
                    curr_group = pe
                elif pe.startswith("-") and "size" not in pe:
                    if not (task == "CLUSTER" and "color" in pe):
                        config = dc(orig_config)
                        config["env_specs"]["task"] = task
                        config["env_specs"]["attendant_edits"] = [{"name": pe[2:]}]
                        if task == "GOTO":
                            config["principal"]["basic_reward_functions"] = [{"name": "reward_reach_object_hof"}]
                            config["attendant"]["basic_reward_functions"] = [{"name": "reward_reach_object_hof"}]
                        elif task == "PICKUP":
                            config["principal"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                            config["attendant"]["basic_reward_functions"] = [{"name": "reward_carry_object_hof"}]
                        elif task in ["PUT", "COLLECT", "CLUSTER"]:
                            config["principal"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                            config["attendant"]["basic_reward_functions"] = [{"name": "reward_adjacent_object_hof"}]
                        if "Island" in curr_group:
                            config["env_specs"]["principal_level"] = "TREASURE_ISLAND"
                            p_level = "TREASURE_ISLAND"
                        else:
                            p_level = "ROOM_DOOR_KEY"
                        test_result = test_control(config, task, p_level = p_level, a_edit = pe[2:])
                        if not test_result:
                            return

    results = np.array(results)
    print("Failure rate:", np.count_nonzero(results == 2) / len(results))
    print("AssertionError rate:", np.count_nonzero(results == 1) / len(results))

    assert sum(results) == 0


def test_control(config, task, p_level = None, a_level = None, a_edit = None):
    global results

    try:
        make_agents(config = config)
        results.append(0)
        return True
    # except (AssertionError, ValueError) as e:
    #     print(f"ASSERT: {task}, principal = {p_level}, attendant = {a_level}")
    #     print(e)
    #     results.append(1)
    except Exception as e:
        if a_level is not None:
            print(f"FAILED: task = {task}, principal = {p_level}, attendant = {a_level}")
        else:
            print(f"FAILED: task = {task}, principal = {p_level}, edit = {a_edit}")
        traceback.print_exc()
        print(e)
        results.append(2)
        return False


if __name__ == "__main__":
     with open("./tests/failed_configs.txt", "w") as f:
        og_stdout = sys.stdout
        og_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        test_construct_all_tasks_and_levels(True)
        sys.stdout = og_stdout
        sys.stderr = og_stderr
