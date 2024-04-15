import sys
sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.basic_utils import xor, convert_to_enum, flatten_list
from package.infrastructure.env_constants import ALLOWABLE_VARIANTS, COLOR_NAMES
from package.enums import Task, Level, Variant
from package.envs.env import *
from package.agents import *
import package.reward_functions as REWARD_FUNCTIONS
from package.envs.env_wrapper import EnvironmentWrapper

from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from typing import List, Dict, Any
import yaml
import random
import time
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


def make_envs(task: Task,
              principal_level: Level,
              attendant_level: Level = None,
              attendant_variants: List[Variant] = None,
              attendant_edits: List[str] = None,
              seed: int = None,
              allowed_object_colors: Union[List, str] = COLOR_NAMES,
              principal_render_mode: str = None,
              attendant_render_mode: str = None,
              principal_setup_actions: List[int] = [],
              attendant_setup_actions: List[int] = []):
    # Some asserts
    assert Task.has_value(task), "Env type is not valid"
    assert Level.has_value(principal_level), "Principal level is not valid"
    assert xor(attendant_level, attendant_variants, attendant_edits) or (not attendant_level and not attendant_variants and not attendant_edits), "Must have only one of `attendant_level`, `attendant_variants`, or `attendant_edits` or none at all"
    if attendant_level:
        assert Level.has_value(attendant_level), "Attendant level is not valid"
    if attendant_variants:
        for sv in attendant_variants:
            assert Variant.has_value(sv), f"Attendant variant \"{sv}\" is not valid"

    # Converting things to enums
    task = convert_to_enum(Task, task)
    principal_level = convert_to_enum(Level, principal_level)
    if attendant_level is not None:
        attendant_level = convert_to_enum(Level, attendant_level)
    if attendant_variants is not None:
        attendant_variants = convert_to_enum(Variant, attendant_variants)

    # Making principal env first
    seed = random.randint(0, 10000) if not seed else seed
    p_env_cls = create_env_class(task, principal_level)
    principal_env = p_env_cls(seed, task, principal_level, render_mode = principal_render_mode, allowed_object_colors = allowed_object_colors)
    
    # Creating the disallowed dictionary for variants
    if attendant_variants is not None:
        disallowed = {}  # FIXME: just revisit some of these
        if Variant.COLOR in attendant_variants:
            disallowed[Variant.COLOR] = principal_env.target_obj.color if hasattr(principal_env, "target_obj") else flatten_list(principal_env.target_objs)[0].color
        if Variant.ROOM_SIZE in attendant_variants:
            disallowed[Variant.ROOM_SIZE] = principal_env.room_size
        if Variant.NUM_OBJECTS in attendant_variants:
            disallowed[Variant.NUM_OBJECTS] = len(principal_env.objs) - 1
        if Variant.OBJECTS in attendant_variants:
            if task in [Task.GOTO, Task.PICKUP]:
                disallowed_objs = [(type(obj[0]), obj[0].color) for obj in principal_env.objs if obj[0] != principal_env.target_obj]
                disallowed_poses = [pos for obj, pos in principal_env.objs if obj != principal_env.target_obj]
                disallowed[Variant.OBJECTS] = (disallowed_objs, disallowed_poses)
            else:
                disallowed_objs = [(type(obj[0]), obj[0].color) for obj in principal_env.objs if obj[0] not in flatten_list(principal_env.target_objs)]
                disallowed_poses = [pos for obj, pos in principal_env.objs if obj not in flatten_list(principal_env.target_objs)]
                disallowed[Variant.OBJECTS] = (disallowed_objs, disallowed_poses)
        if Variant.NUM_ROOMS in attendant_variants:
            disallowed[Variant.NUM_ROOMS] = principal_env.num_rooms if hasattr(principal_env, "num_rooms") else None
        if Variant.ORIENTATION in attendant_variants:
            disallowed[Variant.ORIENTATION] = principal_env
    else:
        disallowed = None

    # Making the attendant env
    target_obj_kwargs = {}
    if task in [Task.GOTO, Task.PICKUP]:
        target_obj_kwargs["target_obj"] = type(principal_env.target_obj)
    elif task == Task.CLUSTER:
        target_obj_kwargs["target_objs"] = [[type(obj) for obj in obj_cluster] for obj_cluster in principal_env.target_objs]
    else:
        target_obj_kwargs["target_objs"] = [type(obj) for obj in principal_env.target_objs]
    if attendant_level:
        a_env_cls = create_env_class(task, attendant_level)
        attendant_env = a_env_cls(seed, task, attendant_level, **target_obj_kwargs, render_mode = attendant_render_mode, allowed_object_colors = allowed_object_colors)
    elif attendant_variants:
        attendant_env = p_env_cls(seed, task, principal_level, **target_obj_kwargs, disallowed = disallowed, render_mode = attendant_render_mode, allowed_object_colors = allowed_object_colors)
    elif attendant_edits:
        attendant_env = copy.deepcopy(principal_env)
        apply_edits(attendant_env, attendant_edits)
    else:
        attendant_env = copy.deepcopy(principal_env)

    # Making environment wrappers
    p_wrapper = EnvironmentWrapper({
        "task": task, "level": principal_level, "render_mode": principal_render_mode
    }, principal_env)
    principal_env.bind_wrapper(p_wrapper)
    a_wrapper = EnvironmentWrapper({
        "task": task, "level": attendant_level, **target_obj_kwargs, "disallowed": disallowed, "render_mode": attendant_render_mode
    }, attendant_env)
    attendant_env.bind_wrapper(a_wrapper)

    # Executing the setup actions, if any
    principal_env.reset()
    for act in principal_setup_actions:
        principal_env.step(act)
    attendant_env.reset()
    for act in attendant_setup_actions:
        attendant_env.step(act)
    
    return principal_env, attendant_env


def create_env_class(task: Task, level: Level):
    class_name = f"{task.value}_{level.value}_Env"
    new_class = type(class_name, (Environment, task_class_mapping[task], level_class_mapping[level]), {"__init__": _custom_init})
    return new_class


def _custom_init(self,
                 env_seed: int,
                 task: Task,
                 level: Level,
                 target_obj: WorldObj = None,
                 target_objs: List[WorldObj] = None,
                 disallowed: Dict[Variant, Any] = None,
                 allowed_object_colors: List[str] = COLOR_NAMES,
                 max_steps: int = None,
                 agent_view_size: int = 5,
                 render_mode = None,
                 **kwargs):
    Environment.__init__(self, env_seed, task, level, target_obj, target_objs, disallowed, allowed_object_colors, max_steps, agent_view_size, render_mode, **kwargs)
    task_cls = task_class_mapping[task]
    level_cls = level_class_mapping[level]
    task_cls.__init__(self)
    level_cls.__init__(self)
    self.initialize_level()
    self._gen_grid(self.room_size, self.room_size)
    self.set_mission()
    # self.set_allowable_skills()
    level_cls.assert_successful_creation(self)


def apply_edits(env: MiniGridEnv, edits: List[Dict]) -> None:
    for parent_class in env.__class__.mro():
        if issubclass(parent_class, BaseLevel) and parent_class is not BaseLevel:
            level_class = parent_class
            break
    invalid_edits = []
    for edit in edits:
        edit_name = edit.pop("name")
        if hasattr(level_class, edit_name):
            edit_method = getattr(env, edit_name)
            edit_method(**edit)
        else:
            invalid_edits.append(edit_name)
    if len(invalid_edits) > 0:
        warnings.warn(f"{len(invalid_edits)} edits could not be applied: {invalid_edits}")


def make_agent(is_principal: bool,
               world_model: MiniGridEnv,
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


def print_allowable_variants(level: Level = None):
    if not level:
        msg = "The following are allowable variants for each level:"
        for level, variants in ALLOWABLE_VARIANTS:
            msg += f"{level.value}: {[v.value for v in variants]}\n"
        print(msg.strip())
    else:
        print(f"The following are allowable variants for level {level.value.upper()}:\n{[v.value for v in ALLOWABLE_VARIANTS[level]]}")


if __name__ == "__main__":
    principal_env, attendant_env = make_envs(task = "CLUSTER",
                                         principal_level = "HIDDEN_KEY",
                                         attendant_variants = ["ORIENTATION"],
                                         seed = 400)
    mc = ManualControl(attendant_env)
    mc.start()
