import sys

sys.path.append("/Users/tutrinh/Work/CHAI/selfexplainingai")

from package.infrastructure.config_utils import ConfigDict
from package.infrastructure.basic_utils import xor, to_enum, flatten_list
from package.infrastructure.env_constants import ALLOWABLE_VARIANTS, COLOR_NAMES
from package.enums import Task, Layout, Edit
from package.envs.env import *
from package.agents import *
import package.reward_functions as REWARD_FUNCTIONS
from package.envs.env_wrapper import EnvironmentWrapper

from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

from typing import List, Dict, Any
import yaml
import random
import time
import inspect
import warnings
import copy


def old_make_agent(
    is_human: bool,
    world_model: MiniGridEnv,
    name: str = None,
    query_source: str = None,
    model_source: str = None,
    basic_reward_functions: List[str] = None,
    basic_reward_weights: List[float] = None,
    skills: List[str] = None,
):
    assert (
        basic_reward_functions is not None
    ), "Must specify `basic_reward_functions` in config"
    assert (
        basic_reward_weights is not None
    ), "Must specify `basic_reward_weights` in config"
    assert len(basic_reward_functions) == len(
        basic_reward_weights
    ), "`basic_reward_functions` and `basic_reward_weights` must be the same length"
    assert skills is not None, "Must specify `skills` in config"

    if is_human:
        agent = Human(query_source, model_source, name=name)
    else:
        agent = AI(query_source, model_source, name=name)
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
        warnings.warn(
            f"{dropped_skills} skills were either not possible or not allowed in this world model for the {'human' if is_human else 'ai'}"
        )

    all_possible_reward_functions = {
        name: func
        for name, func in inspect.getmembers(REWARD_FUNCTIONS, inspect.isfunction)
    }
    for i in range(len(basic_reward_functions)):
        rf = basic_reward_functions[i]
        func_name = rf.pop("name")
        reward_amt = rf.get("amount", 1)
        if func_name in all_possible_reward_functions:
            agent.add_reward_function(
                all_possible_reward_functions[func_name](world_model, amt=reward_amt),
                basic_reward_weights[i],
            )
        else:
            warnings.warn(f"Reward function `{func_name}` is not yet defined")

    return agent


def make_env(config):
    assert Task.has_value(config.task), "Task name {config.task} is invalid!"
    assert Layout.has_value(config.layout), "Layout name {config.layout} is invalid!"
    for edit in config.edits:
        assert Edit.has_value(edit), f"Edit name {edit} is invalid!"

    task = to_enum(Task, config.task)
    layout = to_enum(Layout, config.layout)
    edits = [to_enum(Edit, edit) for edit in config.edits]
    cls = create_env_class(task, layout)
    env = cls(
        config.seed,
        task,
        layout,
        edits,
        allowed_object_colors=config.allowed_object_colors,
        render_mode="human",
    )
    #env = ImgObsWrapper(env)
    return env


def create_env_class(task: Task, layout: Layout):
    class_name = f"{task.value}_{layout.value}_Env"
    new_class = type(
        class_name,
        (
            MindGridEnv,
            task_class_mapping[task],
            layout_class_mapping[layout],
            editor_class_mapping[layout],
        ),
        {"__init__": _custom_init},
    )
    return new_class


def _custom_init(
    self,
    env_seed: int,
    task: Task,
    layout: Layout,
    edits: List[Edit],
    # target_obj: WorldObj = None,
    # target_objs: List[WorldObj] = None,
    # disallowed: Dict[Variant, Any] = None,
    allowed_object_colors: List[str] = COLOR_NAMES,
    # max_steps: int = None,
    # agent_view_size: int = None,
    render_mode=None,
    **kwargs,
):
    MindGridEnv.__init__(
        self,
        env_seed,
        task,
        layout,
        edits,
        # target_obj,
        # target_objs,
        # disallowed,
        allowed_object_colors=allowed_object_colors,
        # max_steps=max_steps,
        # agent_view_size=agent_view_size,
        render_mode=render_mode,
        **kwargs,
    )
    task_class_mapping[task].__init__(self)
    layout_class_mapping[layout].__init__(self)
    editor_class_mapping[layout].__init__(self)

    self._init_task()
    self._init_layout()
    for e in self.edits:
        getattr(self, e.value)()


"""
def make_envs(task: Task,
              human_level: Level,
              ai_level: Level = None,
              ai_variants: List[Variant] = None,
              ai_edits: List[str] = None,
              seed: int = None,
              allowed_object_colors: Union[List, str] = COLOR_NAMES,
              human_render_mode: str = None,
              ai_render_mode: str = None,
              human_setup_actions: List[int] = [],
              ai_setup_actions: List[int] = []):
    # Some asserts
    assert Task.has_value(task), "Env type is not valid"
    assert Level.has_value(human_level), "Human level is not valid"
    assert xor(ai_level, ai_variants, ai_edits) or (not ai_level and not ai_variants and not ai_edits), "Must have only one of `ai_level`, `ai_variants`, or `ai_edits` or none at all"
    if ai_level:
        assert Level.has_value(ai_level), "AI level is not valid"
    if ai_variants:
        for sv in ai_variants:
            assert Variant.has_value(sv), f"AI variant \"{sv}\" is not valid"

    # Converting things to enums
    task = convert_to_enum(Task, task)
    human_level = convert_to_enum(Level, human_level)
    if ai_level is not None:
        ai_level = convert_to_enum(Level, ai_level)
    if ai_variants is not None:
        ai_variants = convert_to_enum(Variant, ai_variants)

    # Making human env first
    seed = random.randint(0, 10000) if not seed else seed
    p_env_cls = create_env_class(task, human_level)
    human_env = p_env_cls(seed, task, human_level, render_mode = human_render_mode, allowed_object_colors = allowed_object_colors)

    # Creating the disallowed dictionary for variants
    if ai_variants is not None:
        disallowed = {}  # FIXME: just revisit some of these
        if Variant.COLOR in ai_variants:
            disallowed[Variant.COLOR] = human_env.target_obj.color if hasattr(human_env, "target_obj") else flatten_list(human_env.target_objs)[0].color
        if Variant.ROOM_SIZE in ai_variants:
            disallowed[Variant.ROOM_SIZE] = human_env.room_size
        if Variant.NUM_OBJECTS in ai_variants:
            disallowed[Variant.NUM_OBJECTS] = len(human_env.objs) - 1
        if Variant.OBJECTS in ai_variants:
            if task in [Task.GOTO, Task.PICKUP]:
                disallowed_objs = [(type(obj[0]), obj[0].color) for obj in human_env.objs if obj[0] != human_env.target_obj]
                disallowed_poses = [pos for obj, pos in human_env.objs if obj != human_env.target_obj]
                disallowed[Variant.OBJECTS] = (disallowed_objs, disallowed_poses)
            else:
                disallowed_objs = [(type(obj[0]), obj[0].color) for obj in human_env.objs if obj[0] not in flatten_list(human_env.target_objs)]
                disallowed_poses = [pos for obj, pos in human_env.objs if obj not in flatten_list(human_env.target_objs)]
                disallowed[Variant.OBJECTS] = (disallowed_objs, disallowed_poses)
        if Variant.NUM_ROOMS in ai_variants:
            disallowed[Variant.NUM_ROOMS] = human_env.num_rooms if hasattr(human_env, "num_rooms") else None
        if Variant.ORIENTATION in ai_variants:
            disallowed[Variant.ORIENTATION] = human_env
    else:
        disallowed = None

    # Making the ai env
    target_obj_kwargs = {}
    if task in [Task.GOTO, Task.PICKUP]:
        target_obj_kwargs["target_obj"] = type(human_env.target_obj)
    elif task == Task.CLUSTER:
        target_obj_kwargs["target_objs"] = [[type(obj) for obj in obj_cluster] for obj_cluster in human_env.target_objs]
    else:
        target_obj_kwargs["target_objs"] = [type(obj) for obj in human_env.target_objs]
    if ai_level:
        a_env_cls = create_env_class(task, ai_level)
        ai_env = a_env_cls(seed, task, ai_level, **target_obj_kwargs, render_mode = ai_render_mode, allowed_object_colors = allowed_object_colors)
    elif ai_variants:
        ai_env = p_env_cls(seed, task, human_level, **target_obj_kwargs, disallowed = disallowed, render_mode = ai_render_mode, allowed_object_colors = allowed_object_colors)
    elif ai_edits:
        ai_env = copy.deepcopy(human_env)
        apply_edits(ai_env, ai_edits)
    else:
        ai_env = copy.deepcopy(human_env)

    # Making environment wrappers
    p_wrapper = EnvironmentWrapper({
        "task": task, "level": human_level, "render_mode": human_render_mode
    }, human_env)
    human_env.bind_wrapper(p_wrapper)
    a_wrapper = EnvironmentWrapper({
        "task": task, "level": ai_level, **target_obj_kwargs, "disallowed": disallowed, "render_mode": ai_render_mode
    }, ai_env)
    ai_env.bind_wrapper(a_wrapper)

    # Executing the setup actions, if any
    human_env.reset()
    for act in human_setup_actions:
        human_env.step(act)
    ai_env.reset()
    for act in ai_setup_actions:
        ai_env.step(act)

    return human_env, ai_env

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


def set_advanced_reward_functions(config_path: str, human: Human, ai: AI):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    assert "human" in config, "Must define a human agent"
    assert "ai" in config, "Must define an ai agent"

    p_kwargs = config["human"]
    a_kwargs = config["ai"]
    if "advanced_reward_functions" in p_kwargs:
        set_advanced_reward_functions_agent(human, p_kwargs)
    if "advanced_reward_functions" in a_kwargs:
        set_advanced_reward_functions_agent(ai, a_kwargs)


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
"""

if __name__ == "__main__":
    human_env, ai_env = make_envs(
        task="CLUSTER", human_level="HIDDEN_KEY", ai_variants=["ORIENTATION"], seed=400
    )
    mc = ManualControl(ai_env)
    mc.start()
