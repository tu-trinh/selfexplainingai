from package.envs import MindGridEnv
from package.envs.modifications import Bridge
from package.infrastructure.env_constants import VEC_TO_DIR, DIR_TO_VEC
from package.infrastructure.env_utils import shortest_path_actions
from package.infrastructure.basic_utils import get_adjacent_cells

from minigrid.core.world_object import WorldObj, Door
from minigrid.core.actions import Actions

from abc import ABC
from gymnasium.core import ActType

from typing import Tuple
from dataclasses import dataclass


class Trajectory:

    def __init__(self, trajectory=None):
        self.states, self.actions = [], []

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_actions(self):
        return len(self.actions)

    def add(self, state: MindGridEnvState, action: ActType = None):
        self.states.append(state)
        if action is not None:
            self.actions.append(action)

    def get(self, i: int) -> Tuple[MindGridEnvState, ActType]:
        if i >= self.n_actions:
            return self.states[i]
        return self.states[i], self.actions[i]

    def slice(self, start: int, end: int):
        assert end >= start, "end must be >= start"
        new_trajectory = Trajectory()
        for i in range(start, end):
            new_trajectory.add(*self.get(i))
        new_trajectory.add(self.get(end))
        self.check(new_trajectory)
        return new_trajectory

    def last_action(self):
        return self.actions[-1]

    def first_state(self):
        return self.states[0]

    def last_state(self):
        return self.states[-1]

    def merge(self, new_trajectory: Trajectory):
        assert (
            self.last_state() == t.first_state()
        ), "Can't merge incompatible trajectories!"
        new_trajectory = self.slice(0, self.n_states - 2)
        for i in range(t.n_states - 1):
            new_trajectory.add(*t.get(i))
        new_trajectory.add(t.last_state(), None)
        self.check(new_trajectory)
        return new_trajectory

    def check(self, t: Trajectory):
        assert t.n_actions == t.n_states - 1


class Skill(ABC):

    def execute(self, env):
        trajectory = Trajectory()
        for a in self.__call__(env):
            if a is None:
                return None
            trajectory.add(env.get_state(), a)
            env.step(a)
        trajectory.append(env.get_state())
        return trajectory

    @abstractmethod
    def __call__(self, env: MindGridEnv):
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def recognize(self, trajectory: List[Tuple[MindGridEnvState, ActType]]):
        raise NotImplementedError


class PrimitiveAction(Skill):

    def __init__(self, action: ActType):
        self.action = action

    def __call__(self, env: MindGridEnv):
        yield self.action

    def recognize(trajectory):
        if trajectory.n_actions > 1:
            return None
        return self.action


class GoTo(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        yield from shortest_path_actions(
            env.gen_simple_2d_map(), start_pos=env.agent_pos, end_pos=[self.pos]
        )

    def recognize(trajectory):
        if trajectory.last_action() != Actions.forward:
            return None
        # actions should all be navigational
        for a in trajectory.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return trajectory.last_state().agent_pos


class GoToAdjacentObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        yield from shortest_path_actions(
            env.gen_simple_2d_map(),
            start_pos=env.agent_pos,
            end_pos=get_adjacent_cells(self.obj.cur_pos, ret_as_list=True),
        )

    def recognize(trajectory):
        # final location must be adjacent to an object
        final_s = trajectory.last_state()
        agent_pos = final_s.agent_pos
        agent_adjacent_cells = get_adjacent_cells(agent_pos)
        adj_obj = None
        for o in final_s.objects:
            if o.cur_pos in agent_adjacent_cells:
                adj_obj = o
                break
        if adj_obj is None:
            return None
        # actions should all be navigational
        for a in trajectory.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return adj_obj


class GoToAdjacentPosition(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        goal = Goal()
        goal.init_pos = self.pos
        yield from GoToAdjacentObject(goal)(env)

    def recognize(trajectory):
        # actions should all be navigational
        for a in trajectory.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return trajectory.last_state().front_pos


class RotateTowardsObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        obj_adjacent_cells = get_adjacent_cells(obj.cur_pos)
        # skill is not applicable if agent is NOT adjacent to an object
        if env.agent_pos not in obj_adjacent_cells:
            return None

        if env.front_pos != obj.cur_pos:
            from_dir = VEC_TO_DIR[tuple(env.dir_vec)]
            dir = (obj.cur_pos[0] - env.agent_pos[0], obj.cur_pos[1] - env.agent_pos[1])
            to_dir = VEC_TO_DIR[dir]

            left_rotations = (to_dir - from_dir + 4) % 4
            right_rotations = (from_dir - to_dir + 4) % 4

            if left_rotations < right_rotations:
                yield from [env.actions.left] * left_rotations
            else:
                yield from [env.actions.right] * right_rotations

    def recognize(trajectory):
        # actions should all be left or right
        for a in trajectory.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        # agent should face towards an object at the end
        final_s = trajectory.last_state()
        fwd_pos = final_s.fwd_pos
        for o in final_s.objects:
            if o.cur_pos == for_pos:
                return o
        return None


class RotateTowardsPosition(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        pos = self.pos

        # applicable only when agent is adjacent to pos
        if env.agent_pos not in get_adjacent_cells(pos):
            return None

        goal = Goal()
        goal.init_pos = pos
        yield from RotateTowardsObject(goal)(env)

    def recognize(trajectory):
        # actions should all be left or right
        for a in trajectory.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        return VEC_TO_DIR[trajectory.last_state().dir_vec]


class DropAt(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        # applicable only when carrying an object
        if env.carrying is not None:
            return None

        yield from GoToAdjacentPosition(self.pos)(env)
        yield env.actions.drop

    def recognize(trajectory):
        # last action must be drop
        if trajectory.last_action() != Actions.drop:
            return None
        # agent should initially carrying an object
        if trajectory.first_state().carrying is None:
            return None
        # in the end, it should not carry any object
        if trajectory.last_state().carrying is not None:
            return None
        return trajectory.first_state().front_pos


class EmptyInventory(Skill):

    def __init__(self):
        pass

    def __call__(self, env: MindGridEnv):
        if env.carrying is None:
            return []

        drop_pos = None
        # find a free adjacent cell
        for pos in get_adjacent_cells(env.agent_pos):
            if env.grid.get(*pos) is None:
                drop_pos = pos
                break

        if drop_pos is None:
            return None

        yield from DropAt(drop_pos)(env)

    def recognize(trajectory):
        return DropAt.recognize(trajectory)


class GetObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # go to object
        yield from GoToAdjacentObject(obj)(env)
        # if agent is carrying an object
        yield from EmptyInventory()(env)
        # rotate towards object
        yield from RotateTowardsObject(obj)(env)
        # if object is hidden in a box, toggle box
        if isinstance(env.grid.get(*obj.cur_pos), Box):
            yield env.actions.toggle
        # else, pick up object
        yield env.actions.pickup

    def recognize(trajectory):
        # last action should be pickup
        if trajectory.last_action() != Actions.pickup:
            return None
        # object carried at the beginning must be different than at the end
        if trajectory.first_state().carrying == trajectory.last_state().carrying:
            return None
        # NOTE: o can be None
        o = trajectory.last_state().carrying
        return o


class MoveObject(Skill):

    def __init__(self, obj: WorldObj, pos: Tuple[int, int]):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # pick up object
        yield from GetObject(obj)(env)
        # drop
        yield from DropAt(pos)

    def recognize(trajectory):
        # last action should be drop
        if trajectory.last_action() != Actions.drop:
            return None
        # = GetObject + DropAt ?
        N = trajectory.n_states
        for i in range(N):
            t1 = trajectory.slice(0, i)
            t2 = trajectory.slice(i, N - 1)
            obj = GetObject.recognize(t1)
            pos = DropAt.recognize(t2)
            if obj is not None and pos is not None:
                return obj, pos
        return None


class GoDirNSteps(Skill):

    def __init__(self, dir: int, n: int):
        self.dir = dir
        self.n = n

    def __call__(self, env: MindGridEnv):
        dir_vec = VEC_TO_DIR[self.dir]
        pos = (env.agent_pos[0] + dir_vec[0], env.agent_pos[1] + dir_vec[1])
        yield from RotateTowardsPosition(goal)(env)
        yield from [env.actions.forward] * self.n

    def recognize(trajectory):
        for i in range(trajectory.n_states):
            t1 = trajectory.slice(0, i)
            dir = RotateTowardsPosition.recognize(t1)
            if dir is None:
                continue
            t2 = trajectory.slice(i, trajectory.n_states - 1)
            # all actions in t2 must be forward
            if t2.actions.count(Actions.forward) != t2.n_actions:
                continue
            return dir, t2.n_actions
        return None


class Unblock(Skill):

    def __init__(self, obj: Door | Bridge):
        assert type(obj) in [Door, Bridge], "Unblock is applicable to only Door or Bridge"
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # check if door is blocked
        adjacent_cells = get_adjacent_cells(obj.init_pos)
        block_obj = None
        for o in env.objects:
            if o in adjacent_cells and o in env.outer_cells:
                block_obj = o
                break

        # if target object is not blocked, do nothing
        if block_obj is None:
            return []

        # otherwise, find a free cell and move blocking object there
        pos = None
        for c in get_adjacent_cells(block_obj):
            if env.grid.get(*c) is None:
                pos = c
                break
        assert pos is not None
        yield from MoveObject(block_obj, pos)(env)

    def recognize(trajectory):

        def get_blocking_object(s, obj):
            blocking_obj = None
            for o in s.objects:
                if o.cur_pos in s.outer_cells and o in get_adjacent_cells(obj.cur_pos):
                    blocking_obj = o
                    break
            return blocking_obj

        # must look like a MoveObject trajectory
        if MoveObject.recognize(trajectory) is None:
            return None

        # find a door or bridge that is initially blocked but is not at the end
        first_s = trajectory.first_state()
        last_s = trajectory.last_state()
        for o in last_s.objects:
            if type(o) in [Door, Bridge]:
                blocking_obj = get_blocking_object(first_s, o)
                if blocking_obj is not None and get_blocking_object(last_s, o) is None:
                    return blocking_obj
        return None


class OpenDoor(Skill):
    """Open a door"""

    def __init__(self, door: Door):
        self.door = door

    def __call__(self, env: MindGridEnv):
        door = self.door

        # if door is already open, do nothing
        if door.is_open:
            return []

        # if door is locked, get key
        if door.is_locked:
            yield from GetObject(env.keys[0])(env)

        # unblock door
        yield from Unblock(door)(env)
        # go to door
        yield from GoToAdjacent(door)(env)
        # toggle door
        yield env.actions.toggle

    def recognize(trajectory):
        # last action must be toggle
        if trajectory.last_action() != Actions.toggle
            return None
        # find a door that is not open initially but is open at the end
        first_s = trajectory.first_state()
        last_s = trajectory.last_state()
        for o in first_s:
            if isinstance(o, Door):
                for oo in last_s:
                    if o == oo and not o.is_open and oo.is_open:
                        return o
        return None


class FixBridge(Skill):
    """Fix a bridge"""

    def __init__(self, bridge: Bridge):
        self.bridge = bridge

    def __call__(self, env: MindGridEnv):
        bridge = self.bridge

        # if bridge is intact, do nothing
        if bridge.is_intact:
            return []

        # get tool
        yield from GetObject(env.tools[0])(env)
        # unblock bridge
        yield from Unblock(bridge)(env)
        # go to bridge
        yield from GoToAdjacent(bridge)(env)
        # toggle bridge
        yield env.actions.toggle

    def recognize(trajectory):
        # last action must be toggle
        if trajectory.last_action() != Actions.toggle
            return None
        # find a bridge that is not intact initially but is intact at the end
        first_s = trajectory.first_state()
        last_s = trajectory.last_state()
        for o in first_s:
            if isinstance(o, Bridge):
                for oo in last_s:
                    if o == oo and not o.is_intact and oo.is_intact:
                        return o
        return None


