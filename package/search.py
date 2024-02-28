from minigrid.core.world_object import Floor, Lava, Goal, Ball, Box, Key, Door, Wall, WorldObj
from minigrid.core.grid import Grid
from minigrid.core.constants import DIR_TO_VEC

from typing import Callable, List, Tuple
from collections import deque
from gymnasium import Env
import copy


class State:
    def __init__(self, loc: Tuple[int, int], dir: int, carrying: WorldObj, grid: Grid):
        self.loc = loc
        self.dir = dir
        self.carrying = carrying
        self.grid = grid
    
    def get_available_actions(self):
        actions = [0, 1, 2]
        dir_vec = DIR_TO_VEC[self.dir]
        in_front = self.grid.get(self.loc[0] + dir_vec[0], self.loc[1] + dir_vec[1])
        if self.carrying is None and type(in_front) in [Ball, Box, Key]:  # can pick object up
            actions.append(3)
        elif self.carrying is not None and type(in_front) in [Floor, Goal]:  # can put object down
            actions.append(4)
        if type(in_front) in [Box, Door]:  # can toggle object
            if isinstance(in_front, Door):
                if in_front.is_locked and isinstance(self.carrying, Key) and self.carrying.color == in_front.color:
                    actions.append(5)
                elif not in_front.is_locked:
                    actions.append(5)
            else:
                actions.append(5)
        if type(in_front) in [Lava, Wall]:  # avoid!
            actions.remove(2)
        return actions


class StateWrapper:
    def __init__(self, env: Env, action_path: List[int]):
        self.env_copy = env
        self.action_path = action_path
        self.state = State(env.agent_pos, env.agent_dir, env.carrying, env.grid)


class Search:
    def __init__(self, type: str, env: Env, goal_check: Callable[[State], bool]):
        self.type = type
        if self.type == "bfs":
            self.fringe = deque()
        self.goal_check = goal_check
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

    def is_goal(self, state: State) -> bool:
        return self.goal_check(state)
    
    def search(self) -> List[int]:
        visited = set()
        while not self.fringe_empty():
            state_wrapper = self.pop_from_fringe()
            state = state_wrapper.state
            action_path = state_wrapper.action_path
            if self.is_goal(state):
                return action_path
            if state not in visited:
                visited.add(state)
                available_actions = state.get_available_actions()
                for aa in available_actions:
                    env_copy = copy.deepcopy(state_wrapper.env_copy)
                    env_copy.step(aa)
                    child = StateWrapper(env_copy, action_path + [aa])
                    self.add_to_fringe(child)
        return None
