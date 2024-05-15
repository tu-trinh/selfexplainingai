import sys
import random

random.seed(2342)

from mindgrid.skills import Skills
from mindgrid.planner import Planner
from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.env_utils import bfs


def is_really_unsolvable(env):
    env.reset()

    if env.keys:
        for d in env.doors:
            d.is_open = True

    if (
        bfs(
            env.gen_simple_2d_map(),
            env.init_agent_dir,
            env.init_agent_pos,
            [env.targets[0].init_pos],
        )
        is None
    ):
        return True

    return False


def execute_plan(plan, env):
    env.reset()
    print()
    for skill, kwargs in plan:
        print(skill, kwargs)
        skill.value(**kwargs)(env)


def plan_and_check(env_config):

    env = make_env(env_config)
    env.reset()
    #env.render()
    #input()

    planner = Planner(env_config)
    skillset = list(Skills)

    plan = planner(env, skillset)

    if plan is None:
        assert is_really_unsolvable(env)
    else:
        execute_plan(plan, env)
        o = env.carrying
        assert o is not None
        assert o.type == "ball" and o.color == env.target_color


def make_config_1(config, seed):
    # go through the passage or open door
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "add_opening",
        "toggle_opening",
        "block_opening",
        "hide_tool_in_box",
        "toggle_opening",
        "remove_tool",
        "add_passage",
    ]
    return env_config


def make_config_2(config, seed):
    # no passage
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "add_opening",
        "toggle_opening",
        "block_opening",
        "hide_tool_in_box",
        "toggle_opening",
        "remove_tool",
    ]

    return env_config

def make_config_3(config, seed):
    # one opening, no passage
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "toggle_opening",
        "block_opening",
        "hide_tool_in_box",
        "toggle_opening",
        "remove_tool",
    ]

    return env_config


def make_config_4(config, seed):
    # one opening, no passage, no block
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "toggle_opening",
        "hide_tool_in_box",
        "toggle_opening",
        "remove_tool",
    ]

    return env_config


def make_config_5(config, seed):
    # one opening, no passage, no block, but with key
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "toggle_opening",
        "hide_tool_in_box",
        "toggle_opening",
    ]

    return env_config


def make_config_6(config, seed):
    # one opening, no passage, no block, but with key, target not in box
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "toggle_opening",
        "hide_tool_in_box",
        "toggle_opening",
    ]

    return env_config


def make_config_7(config, seed):
    # one opening, no passage, no block, but with key, target and tool not in box
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
    ]

    return env_config


def make_config_8(config, seed):
    # no edits
    env_config = config.human.world_model
    env_config.seed = seed
    env_config.edits = []

    return env_config


def make_random_config(config, seed, n=2):
    # randomly choose 2 edits
    env_config = config.human.world_model
    env_config.seed = seed

    all_edits = [
        "double_grid_size",
        "flip_vertical",
        "change_target_color",
        "hide_target_in_box",
        "add_opening",
        "toggle_opening",
        "block_opening",
        "hide_tool_in_box",
        "toggle_opening",
        "remove_tool",
        "add_passage",
    ]

    edits = random.sample(all_edits, n)

    try:
        # double_grid_size must be applied first
        i = edits.index("double_grid_size")
        edits[i], edits[0] = edits[0], edits[i]
    except:
        pass

    print(edits)
    env_config.edits = edits

    return env_config


def test_planner_room_door_key():

    config = make_config(file_path="mindgrid/configs/base.yaml")
    config.human.world_model.layout = "room_door_key"

    for seed in [234]: #[234, 654, 231, 54, 876, 356, 2039, 10392, 6540, 984]:
        print(seed)
        print("TEST 1")
        plan_and_check(make_config_1(config, seed))
        """
        print("TEST 2")
        plan_and_check(make_config_2(config, seed))
        print("TEST 3")
        plan_and_check(make_config_3(config, seed))
        print("TEST 4")
        plan_and_check(make_config_4(config, seed))
        print("TEST 5")
        plan_and_check(make_config_5(config, seed))
        print("TEST 6")
        plan_and_check(make_config_6(config, seed))
        print("TEST 7")
        plan_and_check(make_config_7(config, seed))
        print("TEST 8")
        plan_and_check(make_config_8(config, seed))
        """

    print("TEST RANDOM")
    for _ in range(20):
        seed = random.randint(0, 10000)
        plan_and_check(make_random_config(config, seed))
