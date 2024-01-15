from constants import *
from minigrid.manual_control import ManualControl
from utils import *
from envs.go_to_task import GotoTask
from envs.pick_up_task import PickupTask
from envs.put_next_task import PutTask
from envs.collect_task import CollectTask
from package.enums import *
from typing import List, Dict
import yaml
import random
from agents import Principal, Attendant
import skills as SKILLS
import reward_functions as REWARD_FUNCTIONS


def make_agents(config_path: str):
    with open(config_path, "r") as f:
        config_kwargs = yaml.load(f, Loader = yaml.SafeLoader)
    assert "principal" in config_kwargs, "Must define a principal agent"
    assert "attendant" in config_kwargs, "Must define an attendant agent"
    assert "env_specs" in config_kwargs, "Must define environment specifications"

    principal_env, attendant_env = make_envs(**config_kwargs["env_specs"])
    principal = make_agent(is_principal = True, world_model = principal_env, **config_kwargs["principal"])
    attendant = make_agent(is_principal = False, world_model = attendant_env, **config_kwargs["attendant"])
    return principal, attendant


def make_envs(task: EnvType,
              teacher_level: Level,
              student_level: Level = None,
              student_variants: List[Variant] = None,
              seed: int = None):
    assert EnvType.has_value(task), "Env type is not valid"
    assert Level.has_value(teacher_level), "Teacher level is not valid"
    assert student_level or student_variants, "Must have at least one of `student_level` or `student_variants`"
    if student_level:
        assert Level.has_value(student_level), "Student level is not valid"
    if student_variants:
        for sv in student_variants:
            assert Variant.has_value(sv), f"Student variant \"{sv}\" is not valid"
    
    task = convert_to_enum(task)
    teacher_level = convert_to_enum(teacher_level)
    student_level = convert_to_enum(student_level)
    student_variants = convert_to_enum(student_variants)

    if task == EnvType.GOTO:
        constructor = GotoTask
    elif task == EnvType.PICKUP:
        constructor = PickupTask
    elif task == EnvType.PUT:
        constructor = PutTask
    elif task == EnvType.COLLECT:
        constructor = CollectTask
    
    seed = random.randint(0, 10000) if not seed else seed
    teacher_env = constructor(seed, teacher_level)
    disallowed = {
        Variant.VIEW_SIZE: teacher_env.agent_view_size,
        Variant.COLOR: teacher_env.target_obj.color,
        Variant.ROOM_SIZE: teacher_env.room_size,
        Variant.NUM_OBJECTS: len(teacher_env.objs) - 1,
        Variant.NUM_ROOMS: teacher_env.num_rooms if hasattr(teacher_env, "num_rooms") else None,
    }
    if student_level and student_variants:
        student_env = constructor(seed,
                                  student_level,
                                  target_obj = type(teacher_env.target_obj),
                                  variants = student_variants,
                                  disallowed = disallowed)
    elif student_level:
        student_env = constructor(seed,
                                  student_level,
                                  target_obj = type(teacher_env.target_obj))
    elif student_variants:
        student_env = constructor(seed,
                                  teacher_env.level,
                                  target_obj = type(teacher_env.target_obj),
                                  variants = student_variants,
                                  disallowed = disallowed)

    return teacher_env, student_env


def make_agent(is_principal: bool,
               world_model: gymnasium.Env,
               name: str = None,
               query_source: str = None,
               model_source: str = None,
               reward_functions: List[str] = None,
               reward_weights: List[float] = None,
               skills: List[str] = None):
    assert reward_functions is not None, "Must specify `reward_functions` in config"
    assert reward_weights is not None, "Must specify `reward_weights` in config"
    assert len(reward_functions) == len(reward_weights), "`reward_functions` and `reward_weights` must be the same length"
    assert skills is not None, "Must specify `skills` in config"

    if is_principal:
        agent = Principal(query_source, model_source, name = name)
    else:
        agent = Attendant(query_source, model_source, name = name)
    agent.set_world_model(world_model)
    
    for skill in skills:
        if skill in dir(SKILLS):
            agent.add_skill(skill)
        else:
            raise ValueError(f"Skill `{skill}` is not defined in `skills.py`")
    for i in range(len(reward_functions)):
        if reward_functions[i] in dir(REWARD_FUNCTIONS):
            agent.add_reward_function(reward_functions[i], reward_weights[i])
        else:
            raise ValueError(f"Reward function `{reward_functions[i]}` is not defined in `reward_functions.py`")
    
    return agent


if __name__ == "__main__":
    teacher_env, student_env = make_envs(task = EnvType.GOTO,
                                         teacher_level = Level.HIDDEN_KEY,
                                         student_level = Level.EMPTY,
                                         student_variants = None,
                                         seed = 222)
    mc = ManualControl(teacher_env)
    mc.start()

    # while True:
    #     teacher_env.reset()
    #     time.sleep(1)
    #     teacher_env.step(1)
    #     time.sleep(1)
    #     teacher_env.step(2)
    #     time.sleep(1)