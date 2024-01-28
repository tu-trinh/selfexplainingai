from package.utils import *
from package.constants import *

from minigrid.core.world_object import WorldObj

import gymnasium


"""
Basic reward functions
"""

# Reward for reaching an object
def reward_reach_object_hof(world_model: gymnasium.Env, amt = 1):
    obj_name = OBJ_NAME_MAPPING[type(world_model.target_obj)]
    target_obj_pos = world_model.target_obj_pos

    def reward(env_state: gymnasium.Env, action: int):
        ax, ay = env_state.agent_pos
        tx, ty = target_obj_pos
        if obj_name == "goal":
            if ax == tx and ay == ty:
                return amt
        else:
            if manhattan_distance((ax, ay), (tx, ty)) == 1:
                target_obj_dir = (tx - ax == 1, ty - ay == 1, ax - tx == 1, ay - ty == 1).index(True)
                if env_state.agent_dir == target_obj_dir:
                    return amt
        return 0
    
    reward.__name__ = f"reward_reach_{obj_name}"
    return reward


# Reward for carrying an object
def reward_carry_object_hof(world_model: gymnasium.Env, amt = 1):
    target_obj = world_model.target_obj

    def reward(env_state: gymnasium.Env, action: int):
        if action == env_state.actions.pickup and env_state.carrying and env_state.carrying == target_obj:
            return amt
        return 0
    
    reward.__name__ = f"reward_carry_{OBJ_NAME_MAPPING[type(target_obj)]}"
    return reward


# Reward for objects being adjacent to one another
def reward_adjacent_object_hof(world_model: gymnasium.Env, amt = 1):
    obj_groups = world_model.target_objs
    if not isinstance(obj_groups[0], list):
        obj_groups = [obj_groups]
    obj_group_pos = world_model.target_obj_pos
    if not isinstance(obj_group_pos[0], list):
        obj_group_pos = [obj_group_pos]
    
    def _all_adjacent(positions):
        def _find_path(current, remaining):
            if not remaining:
                return True
            for next_pos in remaining:
                if abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1]) == 1:
                    if _find_path(next_pos, [pos for pos in remaining if pos != next_pos]):
                        return True
            return False
        start_pos = positions[0]
        remaining_positions = positions[1:]
        return _find_path(start_pos, remaining_positions)

    def reward(env_state: gymnasium.Env, action: int):  # FIXME: this seems sus... why not using env state?
        for i in range(len(obj_groups)):
            obj_group = obj_groups[i]
            obj_positions = []
            for j in range(len(obj_group)):
                if obj_group[j].cur_pos is None:
                    obj_positions.append(obj_group_pos[i][j])
                else:
                    obj_positions.append(tuple(obj_group[j].cur_pos))
            if not _all_adjacent(obj_positions):
                return 0
        return amt
    
    reward.__name__ = "reward_adjacent_objects"
    return reward


"""
Advanced reward functions
"""
# Prevent agent from being in or close to a region
# TODO: radial? specific far aways?
def reward_far_away_from_region_hof(world_model: gymnasium.Env, *cells, amt = 1, region_name = "region"):
    assert cells, "At least one cell position must be passed in"

    def reward(env_state: gymnasium.Env, action: int):
        return amt if env_state.agent_pos not in cells else 0
    
    reward.__name__ = "reward_far_away_from_{region_name}"
    return reward


# Encourage agent to go in or near a region
# TODO: radial? specific close tos?
def reward_close_to_region_hof(world_model: gymnasium.Env, *cells, amt = 1, region_name = "region"):
    assert cells, "At least one cell position must be passed in"

    def reward(env_state: gymnasium.Env, action: int):
        return amt if env_state.agent_pos in cells else 0
    
    reward.__name__ = "reward_close_to_{region_name}"
    return reward


# Prefer agent to interact with one object (A) over another (B) of the same type
def reward_objectA_over_objectB(world_model: gymnasium.Env, objA: WorldObj, objB: WorldObj, amt = 1):
    assert type(objA) is type(objB), "Objects must be of the same type"

    def reward(env_state: gymnasium.Env, action: int):
        if action == 3:  # pickup
            if env_state.carrying and env_state.carrying in [objA, objB]:
                return amt if env_state.carrying == objA else 0
        if action == 4 or action == 5:  # drop or toggle
            u, v = env_state.dir_vec
            px, py = env_state.agent_pos[0] + u, env_state.agent_pos[1] + v
            front_obj = env_state.grid.get(px, py)
            if front_obj in [objA, objB]:
                return amt if front_obj == objA else 0
    
    reward.__name__ = f"reward_{objA.color}_{type(objA)}_over_{objB.color}_{type(objB)}"
    return reward
