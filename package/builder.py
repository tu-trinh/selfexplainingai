import sys
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.constants import *
from package.utils import *
from package.enums import *
from package.envs.go_to_task import GotoTask
from package.envs.pick_up_task import PickupTask
from package.envs.put_next_task import PutNextTask
from package.envs.collect_task import CollectTask
from package.envs.cluster_task import ClusterTask
from package.agents import Agent, Principal, Attendant
import package.reward_functions as REWARD_FUNCTIONS

from minigrid.manual_control import ManualControl

from typing import List, Dict
import yaml
import random
import inspect
import warnings


def make_agents(config_path: str = None, config: Dict = None):
    if config is None:
        assert config_path is not None
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader = yaml.SafeLoader)
    assert "principal" in config, "Must define a principal agent"
    assert "attendant" in config, "Must define an attendant agent"
    assert "env_specs" in config, "Must define environment specifications"

    principal_env, attendant_env = make_envs(**config["env_specs"])
    principal = make_agent(is_principal = True, world_model = principal_env, **config["principal"])
    attendant = make_agent(is_principal = False, world_model = attendant_env, **config["attendant"])
    return principal, attendant


def make_envs(task: EnvType,
              principal_level: Level,
              attendant_level: Level = None,
              attendant_variants: List[Variant] = None,
              seed: int = None):
    assert EnvType.has_value(task), "Env type is not valid"
    assert Level.has_value(principal_level), "Teacher level is not valid"
    assert attendant_level or attendant_variants, "Must have at least one of `attendant_level` or `attendant_variants`"
    if attendant_level:
        assert Level.has_value(attendant_level), "Attendant level is not valid"
    if attendant_variants:
        for sv in attendant_variants:
            assert Variant.has_value(sv), f"Attendant variant \"{sv}\" is not valid"

    task = convert_to_enum(EnvType, task)
    principal_level = convert_to_enum(Level, principal_level)
    if attendant_level is not None:
        attendant_level = convert_to_enum(Level, attendant_level)
    if attendant_variants is not None:
        attendant_variants = convert_to_enum(Variant, attendant_variants)

    if task == EnvType.GOTO:
        constructor = GotoTask
    elif task == EnvType.PICKUP:
        constructor = PickupTask
    elif task == EnvType.PUT:
        constructor = PutNextTask
    elif task == EnvType.COLLECT:
        constructor = CollectTask
    elif task == EnvType.CLUSTER:
        constructor = ClusterTask

    seed = random.randint(0, 10000) if not seed else seed
    principal_env = constructor(seed, principal_level)
    if task in [EnvType.GOTO, EnvType.PICKUP]:
        disallowed_objs = set([(type(obj[0]), obj[0].color) for obj in principal_env.objs if obj[0] != principal_env.target_obj])
        disallowed_poses = [pos for obj, pos in principal_env.objs if obj != principal_env.target_obj]
        disallowed_objects = (disallowed_objs, disallowed_poses)
    else:
        disallowed_objs = set([(type(obj[0]), obj[0].color) for obj in principal_env.objs if obj[0] not in flatten_list(principal_env.target_objs)])
        disallowed_poses = [pos for obj, pos in principal_env.objs if obj not in flatten_list(principal_env.target_objs)]
        disallowed_objects = (disallowed_objs, disallowed_poses)
    disallowed = {
        Variant.COLOR: principal_env.target_obj.color if hasattr(principal_env, "target_obj") else flatten_list(principal_env.target_objs)[0].color,
        Variant.ROOM_SIZE: principal_env.room_size,
        Variant.NUM_OBJECTS: len(principal_env.objs) - 1,
        Variant.OBJECTS: disallowed_objects,
        Variant.DOORS: principal_env.doors,
        Variant.NUM_ROOMS: principal_env.num_rooms if hasattr(principal_env, "num_rooms") else None,
        Variant.ORIENTATION: principal_env
    }
    if attendant_level and attendant_variants:
        if task == EnvType.GOTO or task == EnvType.PICKUP:
            attendant_env = constructor(seed, attendant_level, target_obj = type(principal_env.target_obj), variants = attendant_variants, disallowed = disallowed)
        elif task == EnvType.CLUSTER:
            attendant_env = constructor(seed, attendant_level, target_objs = [[type(obj) for obj in obj_cluster] for obj_cluster in principal_env.target_objs], variants = attendant_variants, disallowed = disallowed)
        else:
            attendant_env = constructor(seed, attendant_level, target_objs = [type(obj) for obj in principal_env.target_objs], variants = attendant_variants, disallowed = disallowed)
    elif attendant_level:
        if task == EnvType.GOTO or task == EnvType.PICKUP:
            attendant_env = constructor(seed, attendant_level, target_obj = type(principal_env.target_obj))
        elif task == EnvType.CLUSTER:
            attendant_env = constructor(seed, attendant_level, target_objs = [[type(obj) for obj in obj_cluster] for obj_cluster in principal_env.target_objs], variants = attendant_variants, disallowed = disallowed)
        else:
            attendant_env = constructor(seed, attendant_level, target_objs = [type(obj) for obj in principal_env.target_objs])
    elif attendant_variants:
        if task == EnvType.GOTO or task == EnvType.PICKUP:
            attendant_env = constructor(seed, principal_env.level, target_obj = type(principal_env.target_obj), variants = attendant_variants, disallowed = disallowed)
        elif task == EnvType.CLUSTER:
            attendant_env = constructor(seed, attendant_level, target_objs = [[type(obj) for obj in obj_cluster] for obj_cluster in principal_env.target_objs], variants = attendant_variants, disallowed = disallowed)
        else:
            attendant_env = constructor(seed, principal_env.level, target_objs = [type(obj) for obj in principal_env.target_objs], variants = attendant_variants, disallowed = disallowed)

    return principal_env, attendant_env


def make_agent(is_principal: bool,
               world_model: gymnasium.Env,
               name: str = None,
               query_source: str = None,
               model_source: str = None,
               basic_reward_functions: List[str] = None,
               basic_reward_weights: List[float] = None,
               skills: List[str] = None):
    assert basic_reward_functions is not None, "Must specify `basic_reward_functions` in config"
    assert basic_reward_weights is not None, "Must specify `basic_reward_weights` in config"
    assert len(basic_reward_functions) == len(basic_reward_weights), "`basic_reward_functions` and `basic_reward_weights` must be the same length"
    assert skills is not None, "Must specify `skills` in config"

    if is_principal:
        agent = Principal(query_source, model_source, name = name)
    else:
        agent = Attendant(query_source, model_source, name = name)
    agent.set_world_model(world_model)

    with open("./package/configs/skills.txt", "r") as f:
        all_possible_skills = [s.strip() for s in f.readlines()]
    world_model.set_allowable_skills()
    env_allowable_skills = world_model.allowable_skills
    dropped_skills = 0
    for skill in skills:
        if skill in all_possible_skills and skill in env_allowable_skills.keys():
            agent.add_skill(skill)
        else:
            dropped_skills += 1
    if dropped_skills > 0:
        warnings.warn(f"{dropped_skills} skills were either not possible or not allowed in this world model for the {'principal' if is_principal else 'attendant'}")

    all_possible_reward_functions = {name: func for name, func in inspect.getmembers(REWARD_FUNCTIONS, inspect.isfunction)}
    for i in range(len(basic_reward_functions)):
        rf = basic_reward_functions[i]
        func_name = rf.pop("name")
        reward_amt = rf.get("amount", 1)
        if func_name in all_possible_reward_functions:
            agent.add_reward_function(all_possible_reward_functions[func_name](world_model, amt = reward_amt), basic_reward_weights[i])
        else:
            warnings.warn(f"Reward function `{func_name}` is not yet defined")

    return agent


def set_advanced_reward_functions(config_path: str, principal: Principal, attendant: Attendant):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    assert "principal" in config, "Must define a principal agent"
    assert "attendant" in config, "Must define an attendant agent"

    p_kwargs = config["principal"]
    a_kwargs = config["attendant"]
    if "advanced_reward_functions" in p_kwargs:
        set_advanced_reward_functions_agent(principal, p_kwargs)
    if "advanced_reward_functions" in a_kwargs:
        set_advanced_reward_functions_agent(attendant, a_kwargs)


def set_advanced_reward_functions_agent(agent: Agent, agent_kwargs: Dict):
    assert "advanced_reward_weights" in agent_kwargs, "Must define `advanced_reward_weights` for `advanced_reward_functions`"
    assert len(agent_kwargs["advanced_reward_functions"]) == len(agent_kwargs["advanced_reward_weights"]), "`advanced_reward_functions` and `advanced_reward_weights` must be the same length"

    all_possible_reward_functions = {name: func for name, func in inspect.getmembers(REWARD_FUNCTIONS, inspect.isfunction)}
    reward_functions = agent_kwargs["advanced_reward_functions"]
    reward_weights = agent_kwargs["advanced_reward_weights"]
    for i in range(len(reward_functions)):
        rf = reward_functions[i]
        if rf["name"] in all_possible_reward_functions:
            func_name = rf.pop("name")
            func_args = rf
            agent.add_reward_function(all_possible_reward_functions[func_name](world_model = agent.world_model, **func_args), reward_weights[i])
        else:
            raise ValueError(f"Reward function `{rf['name']}` is not yet defined")


if __name__ == "__main__":
    principal_env, attendant_env = make_envs(task = "CLUSTER",
                                         principal_level = "HIDDEN_KEY",
                                         attendant_variants = ["ORIENTATION"],
                                         seed = 400)
    mc = ManualControl(attendant_env)
    mc.start()

    # while True:
    #     principal_env.reset()
    #     time.sleep(1)
    #     principal_env.step(1)
    #     time.sleep(1)
    #     principal_env.step(2)
    #     time.sleep(1)
