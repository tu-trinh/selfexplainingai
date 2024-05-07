from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from minigrid.core.actions import Actions
from minigrid.core.world_object import Box, Goal

from mindgrid.envs.objects import Bridge, DoorWithDirection
from mindgrid.infrastructure.basic_utils import get_adjacent_cells
from mindgrid.infrastructure.env_constants import VEC_TO_DIR
from mindgrid.infrastructure.env_utils import bfs
from mindgrid.infrastructure.trajectory import Trajectory


class Skill(ABC):

    @staticmethod
    def execute(env: MindGridEnv, actions: List[ActType]) -> Trajectory:
        if actions is None:
            return None
        t = Trajectory()
        env.render()
        input()
        for a in actions:
            t.add(env.get_state(), a)
            env.step(a)
            env.render()
            input()
        t.add(env.get_state())
        return t

    @staticmethod
    def execute_and_merge(
        t: Trajectory, env: MindGridEnv, actions: List[ActType]
    ) -> Trajectory:
        tt = Skill.execute(env, actions)
        if tt is None:
            return None
        return t.merge(tt)

    @abstractmethod
    def __call__(self, env: MindGridEnv):
        raise NotImplementedError

    @abstractmethod
    def recognize(self, t: Trajectory):
        raise NotImplementedError


class PrimitiveAction(Skill):

    def __init__(self, action: ActType):
        self.action = action

    def __call__(self, env: MindGridEnv):
        return Skill.execute(env, [self.action])

    def recognize(t: Trajectory):
        if t.n_actions > 1:
            return None
        return self.action


class GoTo(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        actions = bfs(
            env.gen_simple_2d_map(),
            env.agent_dir,
            env.agent_pos,
            [self.pos],
        )
        return Skill.execute(env, actions)

    def recognize(t):
        if t.last_action != Actions.forward:
            return None
        # actions should all be navigational
        for a in t.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return t.last_state.agent_pos


class RotateTowardsObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # skill is not applicable if agent is NOT adjacent to an object
        if env.agent_pos not in get_adjacent_cells(obj.cur_pos):
            return None

        from_dir = VEC_TO_DIR[tuple(env.dir_vec)]
        dir = (obj.cur_pos[0] - env.agent_pos[0], obj.cur_pos[1] - env.agent_pos[1])
        to_dir = VEC_TO_DIR[dir]

        left_rotations = (from_dir - to_dir + 4) % 4
        right_rotations = (to_dir - from_dir + 4) % 4

        if left_rotations < right_rotations:
            actions = [env.actions.left] * left_rotations
        else:
            actions = [env.actions.right] * right_rotations
        return Skill.execute(env, actions)

    def recognize(t):
        # actions should all be left or right
        for a in t.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        # agent should face towards an object at the end
        final_s = t.last_state
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
        goal.init_pos = goal.cur_pos = pos
        return RotateTowardsObject(goal)(env)

    def recognize(t):
        # actions should all be left or right
        for a in t.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        return VEC_TO_DIR[t.last_state.dir_vec]


class GoToAdjacentObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj
        actions = bfs(
            env.gen_simple_2d_map(),
            env.agent_dir,
            env.agent_pos,
            get_adjacent_cells(obj.cur_pos, ret_as_list=True),
        )
        t = Skill.execute(env, actions)
        t = t.merge(RotateTowardsObject(obj)(env))
        return t

    def recognize(t):
        # final location must be adjacent to an object
        final_s = t.last_state
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
        for a in t.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return adj_obj


class GoToAdjacentPosition(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        goal = Goal()
        goal.init_pos = goal.cur_pos = self.pos
        return GoToAdjacentObject(goal)(env)

    def recognize(t):
        # actions should all be navigational
        for a in t.actions():
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return t.last_state.front_pos


class DropAt(Skill):

    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        # applicable only when carrying an object
        if env.carrying is None:
            return None
        t = GoToAdjacentPosition(self.pos)(env)
        t = t.merge(Skill.execute(env, [env.actions.drop]))
        return t

    def recognize(t):
        # last action must be drop
        if t.last_action != Actions.drop:
            return None
        # agent should initially carrying an object
        if t.first_state.carrying is None:
            return None
        # in the end, it should not carry any object
        if t.last_state.carrying is not None:
            return None
        return t.first_state.front_pos


class EmptyInventory(Skill):

    def __init__(self):
        pass

    def __call__(self, env: MindGridEnv):

        # if inventory is empty, do nothing
        if env.carrying is None:
            return Skill.execute(env, [])

        # find a free adjacent cell to drop
        # FIXME: should search over a larger area
        drop_pos = None
        for pos in get_adjacent_cells(env.agent_pos):
            if env.grid.get(*pos) is None:
                drop_pos = pos
                break

        if drop_pos is None:
            return None

        return DropAt(drop_pos)(env)

    def recognize(t):
        return DropAt.recognize(t)


class GetObject(Skill):

    def __init__(self, obj: WorldObj):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj
        # if agent is carrying an object, drop it
        t = EmptyInventory()(env)
        # go to object
        t = t.merge(GoToAdjacentObject(obj)(env))
        # if object is hidden in a box, toggle box
        if isinstance(env.grid.get(*obj.cur_pos), Box):
            t.merge(Skill.execute(env, [env.actions.toggle]))
        # pick up object
        t.merge(Skill.execute(env, [env.actions.pickup]))
        return t

    def recognize(t):
        # last action should be pickup
        if t.last_action != Actions.pickup:
            return None
        # object carried at the beginning must be different than at the end
        if t.first_state.carrying == t.last_state.carrying:
            return None
        # NOTE: o can be None
        o = t.last_state.carrying
        return o


class MoveObject(Skill):

    def __init__(self, obj: WorldObj, pos: Tuple[int, int]):
        self.obj = obj
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        obj, pos = self.obj, self.pos

        # pick up object
        t = GetObject(obj)(env)
        # drop
        t = t.merge(DropAt(pos)(env))
        return t

    def recognize(t):
        # last action should be drop
        if t.last_action != Actions.drop:
            return None
        # = GetObject + DropAt ?
        N = t.n_states
        for i in range(N):
            t1 = t.slice(0, i)
            t2 = t.slice(i, N - 1)
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
        t = Trajectory()
        # rotate to dir
        dir_vec = VEC_TO_DIR[self.dir]
        pos = (env.agent_pos[0] + dir_vec[0], env.agent_pos[1] + dir_vec[1])
        t = t.merge(RotateTowardsPosition(pos)(env))
        # move forward n steps
        t = t.merge(Skill.execute(env, [env.actions.forward] * self.n))
        return t

    def recognize(t):
        for i in range(t.n_states):
            t1 = t.slice(0, i)
            dir = RotateTowardsPosition.recognize(t1)
            if dir is None:
                continue
            t2 = t.slice(i, t.n_states - 1)
            # all actions in t2 must be forward
            if t2.actions.count(Actions.forward) != t2.n_actions:
                continue
            return dir, t2.n_actions
        return None


class Unblock(Skill):

    def __init__(self, obj: DoorWithDirection | Bridge):
        assert obj.type in [
            "door",
            "bridge",
        ], "Unblock is applicable to only Door or Bridge"
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj
        # check if opening is blocked
        adjacent_cells = get_adjacent_cells(obj.init_pos)
        block_obj = None
        for o in env.objects:
            if o.cur_pos in adjacent_cells and o.cur_pos in env.outer_cells:
                block_obj = o
                break

        # if target object is not blocked, do nothing
        if block_obj is None:
            return Skill.execute(env, [])

        # otherwise, find a free cell and move blocking object there
        # FIXME: should search in a larger area
        pos = None
        for c in get_adjacent_cells(block_obj.cur_pos):
            if env.grid.get(*c) is None:
                pos = c
                break

        if pos is None:
            return None

        return MoveObject(block_obj, pos)(env)

    def recognize(t):

        def get_blocking_object(s, obj):
            blocking_obj = None
            for o in s.objects:
                if o.cur_pos in s.outer_cells and o in get_adjacent_cells(obj.cur_pos):
                    blocking_obj = o
                    break
            return blocking_obj

        # must look like a MoveObject t
        if MoveObject.recognize(t) is None:
            return None

        # find a door or bridge that is initially blocked but is not at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in last_s.objects:
            if o.type in ["door", "bridge"]:
                blocking_obj = get_blocking_object(first_s, o)
                if blocking_obj is not None and get_blocking_object(last_s, o) is None:
                    return blocking_obj
        return None


class OpenBox(Skill):

    def __init__(self, box: Box):
        assert box.type == "box", "OpenBox is applicable to only boxes"
        self.box = box

    def __call__(self, env: MindGridEnv):
        box = self.box
        # go to the box and toggle it
        t = GoToAdjacentObject(box)(env)
        t = t.merge(Skill.execute(env, [env.actions.toggle]))
        return t

    def recognize(self, t):
        if t.last_action != Actions.toggle:
            return None
        # agent must face towards a box
        last_s = t.last_state
        for o in last_s.objects:
            if o.type == "box" and o.cur_pos == last_s.front_pos:
                return o
        return None


class OpenDoor(Skill):
    """Open a door"""

    def __init__(self, door: DoorWithDirection):
        self.door = door

    def __call__(self, env: MindGridEnv):
        door = self.door

        # if door is already open, do nothing
        if door.is_open:
            return Skill.execute(env, [])

        # unblock door
        t = Unblock(door)(env)

        print(env.keys)

        # if door is locked, get key
        if door.is_locked:
            # if there is no key, can't open
            if not env.keys:
                return None
            t = t.merge(GetObject(env.keys[0])(env))

        # go to door
        t = t.merge(GoToAdjacentObject(door)(env))
        # toggle door
        t = t.merge(self.execute(env, [env.actions.toggle]))
        return t

    def recognize(t):
        # last action must be toggle
        if t.last_action != Actions.toggle:
            return None
        # find a door that is not open initially but is open at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in first_s.objects:
            if o.type == "door":
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
            return Skill.execute(env, [])

        # if there is no tool, can't fix
        if not env.tools:
            return None

        # unblock bridge
        t = Unblock(bridge)(env)
        # get tool
        t = t.merge(GetObject(env.tools[0])(env))
        # go to bridge
        t = t.merge(GoToAdjacentObject(bridge)(env))
        # toggle bridge
        t = t.merge(Skill.execute(env, [env.actions.toggle]))
        return t

    def recognize(t):
        # last action must be toggle
        if t.last_action != Actions.toggle:
            return None
        # find a bridge that is not intact initially but is intact at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in first_s.objects:
            if o.type == "bridge":
                for oo in last_s:
                    if o == oo and not o.is_intact and oo.is_intact:
                        return o
        return None
