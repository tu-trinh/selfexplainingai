from __future__ import annotations

from package.infrastructure.basic_utils import debug, compare_world_obj
from package.infrastructure.env_utils import get_obs_desc
from package.infrastructure.env_constants import OBJ_NAME_MAPPING, IDX_TO_ACTION, DIR_TO_VEC

from minigrid.core.world_object import Floor, Lava, Goal, Ball, Box, Key, Door, Wall, WorldObj
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv

from typing import Callable, List, Tuple, Union
from collections import deque
from gymnasium import Env
import copy


class State:
    def __init__(self, loc: Tuple[int, int], dir: int, carrying: WorldObj, grid: Grid):
        self.loc = loc
        self.dir = dir
        self.carrying = carrying
        self.grid = grid
    
    def __eq__(self, other: State):
        non_grid_components = [
            self.loc == other.loc,
            self.dir == other.dir,
            (self.carrying is None and other.carrying is None) or compare_world_obj(self.carrying, other.carrying)
        ]
        return all(non_grid_components) and self.grid == other.grid
        # grid_components = [compare_world_obj(self.grid.grid[i], self.other.grid[i]) for i in range(len(self.grid.grid))]
        # return all(non_grid_components) and all(grid_components)
    
    def hash_helper(self, obj: WorldObj):
        if obj is None:
            return (None, None, None)
        if type(obj) == Floor:  # don't think this is needed...
            return ("floor", "grey")
        if type(obj) == Door:
            return ("door", obj.color, obj.init_pos, obj.is_locked, obj.is_open)
        if type(obj) == Box:
            return ("box", obj.color, obj.init_pos, self.hash_helper(obj.contains))
        return (OBJ_NAME_MAPPING[type(obj)], obj.color, obj.init_pos)
    
    def __hash__(self):
        agent_components = (self.loc, self.dir)
        carrying_components = self.hash_helper(self.carrying)
        grid_components = tuple(self.hash_helper(obj) for obj in self.grid.grid)
        return hash((agent_components, carrying_components, grid_components))
    
    def get_available_actions(self):
        actions = [0, 1, 2]
        dir_vec = DIR_TO_VEC[self.dir]
        in_front = self.grid.get(self.loc[0] + dir_vec[0], self.loc[1] + dir_vec[1])
        if self.carrying is None and type(in_front) in [Ball, Box, Key]:  # can pick object up
            actions.append(3)
        elif self.carrying is not None and type(in_front) in [type(None), Goal]:  # can put object down
            actions.append(4)
        if type(in_front) in [Box, Door]:  # can toggle object
            if isinstance(in_front, Door):
                if in_front.is_locked and isinstance(self.carrying, Key) and self.carrying.color == in_front.color:
                    actions.append(5)
                elif not in_front.is_locked:
                    actions.append(5)
            else:
                actions.append(5)
        if (in_front is not None) and (not type(in_front) in [Goal]) and not (type(in_front) == Door and in_front.is_open):  # can't move forward
            actions.remove(2)
        return actions


class StateWrapper:
    def __init__(self, env: Env, action_path: List[int]):
        self.env_copy = env
        self.action_path = action_path
        self.state = State(env.agent_pos, env.agent_dir, env.carrying, env.grid)


class Search:
    def __init__(self, type: str, env: Env, goal_check: Callable[[Union[State, MiniGridEnv]], bool], check_unit: str):
        self.type = type
        if self.type == "bfs":
            self.fringe = deque()
        self.goal_check = goal_check
        self.check_unit = check_unit  # does goal check take in "s" state or "e" environment
        start_state_wrapper = StateWrapper(env, [])
        self.add_to_fringe(start_state_wrapper)
    
    def fringe_empty(self) -> bool:
        return len(self.fringe) == 0
    
    def add_to_fringe(self, state_wrapper: StateWrapper) -> None:
        if self.type == "bfs":
            self.fringe.append(state_wrapper)
    
    def pop_from_fringe(self) -> StateWrapper:
        if self.type == "bfs":
            return self.fringe.popleft()

    def is_goal(self, state: Union[State, MiniGridEnv]) -> bool:
        return self.goal_check(state)
    
    def search(self, testing = False) -> List[int]:
        if testing:
            visited = set()
            first_state_wrapper = self.pop_from_fringe()
            first_state = first_state_wrapper.state
            visited.add(first_state)
            debug("set size", len(visited))
            visited.add(first_state)
            debug("set size", len(visited))

            env_copy = copy.deepcopy(first_state_wrapper.env_copy)
            env_copy.step(2)
            env_copy.step(1)
            env_copy.step(3)
            child = StateWrapper(env_copy, [])
            assert first_state != child.state, "eq failed"
            assert hash(first_state) != hash(child.state), "hash failed"
            visited.add(child.state)
            debug("set size", len(visited))

            env_copy = copy.deepcopy(child.env_copy)
            env_copy.step(4)
            env_copy.step(1)
            env_copy.step(0)
            env_copy.step(3)
            grandchild = StateWrapper(env_copy, [])
            assert child.state == grandchild.state, "eq failed"
            assert hash(child.state) == hash(grandchild.state), "hash failed"
            visited.add(grandchild.state)
            debug("set size", len(visited))
            return
        visited = set()
        # fuck_env = copy.deepcopy(self.fringe[0].env_copy)
        # for shit in [0, 3, 1, 4, 0, 2, 1, 2, 2, 2, 0, 3, 0, 2, 2, 2, 1, 5, 1]:
            # fuck_env.step(shit)
        # fuck_wrapper = StateWrapper(fuck_env, [])
        while not self.fringe_empty():
            state_wrapper = self.pop_from_fringe()
            state = state_wrapper.state
            action_path = state_wrapper.action_path
            if self.check_unit == "s":
                if self.is_goal(state):
                    # debug("STATE IS A GOAL")
                    return action_path
            else:
                if self.is_goal(state_wrapper.env_copy):
                    # debug("STATE IS A GOAL")
                    return action_path
            if state not in visited:
                visited.add(state)
                # if hash(state) == hash(fuck_wrapper.state):
                    # debug("THAT'S THE BITCH")
                    # debug([IDX_TO_ACTION[act] for act in action_path])
                # if len(action_path) == 0:
                    # debug("START STATE")
                # else:
                    # debug("ACTIONS NEEDED TO REACH BELOW STATE")
                    # debug([IDX_TO_ACTION[act] for act in action_path])
                available_actions = state.get_available_actions()
                for aa in available_actions:
                    env_copy = copy.deepcopy(state_wrapper.env_copy)
                    env_copy.step(aa)
                    child = StateWrapper(env_copy, action_path + [aa])
                    self.add_to_fringe(child)
            # else:
                # if hash(state) == hash(fuck_wrapper.state):
                    # debug("repeated state with action sequence")
                    # debug([IDX_TO_ACTION[act] for act in action_path])
        return None
