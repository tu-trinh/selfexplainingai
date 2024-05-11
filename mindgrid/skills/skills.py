from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from minigrid.core.actions import Actions
from minigrid.core.world_object import Box, Goal

from mindgrid.envs.objects import Bridge, DoorWithDirection
from mindgrid.infrastructure.basic_utils import get_adjacent_cells, CustomEnum
from mindgrid.infrastructure.env_constants import VEC_TO_DIR, DIR_TO_VEC
from mindgrid.infrastructure.env_utils import bfs
from mindgrid.infrastructure.trajectory import Trajectory, NullTrajectory


def execute(env: MindGridEnv, actions: List[ActType], render=False) -> Trajectory:
    if actions is None:
        return None
    t = Trajectory()
    if render:
        env.render()
    for a in actions:
        t.add(env.get_state(), a)
        env.step(a)
        if render:
            env.render()
    t.add(env.get_state())
    return t


def _get_same_object(s: MindGridEnvState, o: WorldObj) -> WorldObj:
    for oo in s.objects:
        if oo.init_pos == o.init_pos and oo.type == o.type:
            return oo
    return None


def _find_free_cell(env: MindGridEnv, start_pos: Tuple[int, int], r=1) -> Tuple[int, int]:
    cand = []
    for dx in range(-r, r):
        for dy in range(-r, r):
            if dx == dy == 0:
                continue
            pos = start_pos[0] + dx, start_pos[1] + dy
            if (
                0 <= pos[0] < env.width
                and 0 <= pos[1] < env.height
                and env.grid.get(*pos) is None
            ):
                cand.append((abs(dx) + abs(dy), pos))
    if cand:
        return sorted(cand, key=lambda x: x[0])[0][1]
    return None


class BaseSkill(ABC):

    @abstractmethod
    def __call__(self, env: MindGridEnv):
        raise NotImplementedError

    @abstractmethod
    def recognize(self, t: Trajectory):
        raise NotImplementedError


class Primitive(BaseSkill):

    def __init__(self, action: ActType = None):
        self.action = action

    def __call__(self, env: MindGridEnv):
        return execute(env, [self.action])

    def recognize(t: Trajectory):
        if t.n_actions != 1:
            return None
        return {"action": t.last_action}


class GoTo(BaseSkill):

    def __init__(self, pos: Tuple[int, int] = None):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        actions = bfs(
            env.gen_simple_2d_map(),
            env.agent_dir,
            env.agent_pos,
            [self.pos],
        )
        if actions is None:
            return NullTrajectory()
        return execute(env, actions)

    def recognize(t):
        if t.n_actions == 0:
            return {"pos": t.last_state.agent_pos}
        # last action must be forward
        if t.last_action != Actions.forward:
            return None
        # actions should all be navigational
        for a in t.actions:
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None

        pos = t.last_state.agent_pos

        first_s = t.first_state
        actions = bfs(
            first_s.simple_2d_map,
            first_s.agent_dir,
            first_s.agent_pos,
            [pos],
        )

        if actions == t.actions:
            return {"pos": pos}

        return None


class RotateTowardsObject(BaseSkill):

    def __init__(self, obj: WorldObj = None):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # skill is not applicable if agent is NOT adjacent to an object
        if env.agent_pos not in get_adjacent_cells(obj.cur_pos):
            return NullTrajectory()

        from_dir = VEC_TO_DIR[tuple(env.dir_vec)]
        dir = (obj.cur_pos[0] - env.agent_pos[0], obj.cur_pos[1] - env.agent_pos[1])
        to_dir = VEC_TO_DIR[dir]

        left_rotations = (from_dir - to_dir + 4) % 4
        right_rotations = (to_dir - from_dir + 4) % 4

        if left_rotations < right_rotations:
            actions = [env.actions.left] * left_rotations
        else:
            actions = [env.actions.right] * right_rotations
        return execute(env, actions)

    def recognize(t):
        # actions should all be left or right
        for a in t.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        # agent should face towards an object at the end
        last_s = t.last_state
        for o in last_s.objects:
            if o.cur_pos == last_s.front_pos:
                return {"obj": o}
        return None


class RotateTowardsDirection(BaseSkill):

    def __init__(self, dir: int = None):
        self.dir = dir

    def __call__(self, env: MindGridEnv):
        dir_vec = DIR_TO_VEC[self.dir]
        goal = Goal()
        goal.init_pos = goal.cur_pos = (
            env.agent_pos[0] + dir_vec[0],
            env.agent_pos[1] + dir_vec[1],
        )
        return RotateTowardsObject(goal)(env)

    def recognize(t):
        # actions should all be left or right
        for a in t.actions:
            if a not in [Actions.left, Actions.right]:
                return None
        return {"dir": VEC_TO_DIR[t.last_state.dir_vec]}


class GoToAdjacentObject(BaseSkill):

    def __init__(self, obj: WorldObj = None):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # NOTE: there are at most 4 positions that are adjacent to an object
        # we want to make the final state deterministic
        # hence, we select the position that yield the shortest sequence of actions
        cand = []
        for dir_vec in VEC_TO_DIR:
            actions = bfs(
                env.gen_simple_2d_map(),
                env.agent_dir,
                env.agent_pos,
                [(obj.cur_pos[0] + dir_vec[0], obj.cur_pos[1] + dir_vec[1])],
            )
            if actions is not None:
                cand.append(actions)
        if not cand:
            return NullTrajectory()

        cand = sorted(cand, key=lambda x: len(x))
        t = execute(env, cand[0])
        t = t.merge(RotateTowardsObject(obj)(env))
        return t

    def recognize(t):
        # actions should all be navigational
        for a in t.actions:
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None

        # agent must face towards an object
        last_s = t.last_state
        obj = None
        for o in last_s.objects:
            if o.cur_pos == last_s.front_pos:
                obj = o
                break

        if obj is None:
            return None

        # check if the agent's final position yields the shortest path from the initial position
        # among 4 positions that are adjacent to the object
        first_s = t.first_state
        cand = []
        for dir_vec in VEC_TO_DIR:
            pos = (obj.cur_pos[0] + dir_vec[0], obj.cur_pos[1] + dir_vec[1])
            actions = bfs(
                first_s.simple_2d_map,
                first_s.agent_dir,
                first_s.agent_pos,
                [pos],
            )
            if actions is not None:
                cand.append((len(actions), pos))
        cand = sorted(cand, key=lambda x: x[0])
        # position that yield shortest path must be agent's last position
        if cand[0][1] == last_s.agent_pos:
            return {"obj": obj}

        return None


class GoToAdjacentPosition(BaseSkill):

    def __init__(self, pos: Tuple[int, int] = None):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        goal = Goal()
        goal.init_pos = goal.cur_pos = self.pos
        return GoToAdjacentObject(goal)(env)

    def recognize(t):
        # actions should all be navigational
        for a in t.actions:
            if a not in [Actions.left, Actions.right, Actions.forward]:
                return None
        return {"pos": t.last_state.front_pos}


class DropAt(BaseSkill):

    def __init__(self, pos: Tuple[int, int] = None):
        self.pos = pos

    def __call__(self, env: MindGridEnv):
        # applicable only when carrying an object
        if env.carrying is None:
            return NullTrajectory()
        # go to pos and drop
        t = GoToAdjacentPosition(self.pos)(env)
        t = t.merge(execute(env, [env.actions.drop]))
        return t

    def recognize(t):
        # last action must be drop
        if t.n_actions == 0 or t.last_action != Actions.drop:
            return None
        # agent should initially carrying an object
        if t.first_state.carrying is None:
            return None
        # first part must look like GoToAdjacentPosition
        pos = GoToAdjacentPosition.recognize(t.slice(0, t.n_states - 2))
        if pos is None:
            return None
        return pos


class EmptyInventory(BaseSkill):

    def __init__(self):
        pass

    def __call__(self, env: MindGridEnv):

        # if inventory is empty, do nothing
        if env.carrying is None:
            return execute(env, [])

        # find a free adjacent cell to drop
        drop_pos = _find_free_cell(env, env.agent_pos)

        if drop_pos is None:
            return NullTrajectory()

        return DropAt(drop_pos)(env)

    def recognize(t):
        if t.n_actions == 0:
            o = t.first_state.carrying
            # inventory is empty
            if o is None:
                return {}
            # inventory is non-empty
            return None
        if DropAt.recognize(t) is None:
            return None
        return {}


class OpenBox(BaseSkill):

    def __init__(self, box: Box = None):
        assert box.type == "box", "OpenBox is applicable to only boxes"
        self.box = box

    def __call__(self, env: MindGridEnv):
        box = self.box
        # box has been opened, skill is not applicable
        o = env.grid.get(*box.cur_pos)
        if o.type != "box":
            return NullTrajectory()
        # go to the box and toggle it
        t = GoToAdjacentObject(box)(env)
        t = t.merge(execute(env, [env.actions.toggle]))
        return t

    def recognize(t):
        if t.n_actions == 0:
            return None
        if t.last_action != Actions.toggle:
            return None
        # first part must look like GoToAdjacentObject
        ret = GoToAdjacentObject.recognize(t.slice(0, t.n_states - 2))
        if ret is None:
            return None
        for o in t.first_state.objects:
            if o.type == "box" and o.cur_pos == t.last_state.front_pos:
                return {"box": o}
        return None


class GetObject(BaseSkill):

    def __init__(self, obj: WorldObj = None):
        self.obj = obj

    def __call__(self, env: MindGridEnv):
        obj = self.obj

        # already have object, do nothing
        if env.carrying == obj:
            return execute(env, [])

        # can't get next to object
        if (
            bfs(
                env.gen_simple_2d_map(),
                env.agent_dir,
                env.agent_pos,
                get_adjacent_cells(obj.cur_pos),
            )
            is None
        ):
            return NullTrajectory()
        # if object is in a box, open it
        o = env.grid.get(*obj.cur_pos)
        if isinstance(o, Box):
            t = OpenBox(o)(env)
        else:
            # else, go to object
            t = GoToAdjacentObject(obj)(env)
        # if agent is carrying an object, drop it
        t = t.merge(EmptyInventory()(env))
        # rotate towards object
        t = t.merge(RotateTowardsObject(obj)(env))
        # pick up object
        t = t.merge(execute(env, [env.actions.pickup]))
        return t

    def recognize(t):
        # last action should be pickup
        if t.n_actions == 0 or t.actions.count(Actions.pickup) != 1:
            return None
        # must carry an object at the end
        if t.last_state.carrying is None:
            return None

        N = t.n_states - 1
        for i in range(N):
            t1 = t.slice(0, i)
            if (
                OpenBox.recognize(t1) is not None
                or GoToAdjacentObject.recognize(t1) is not None
            ):
                for j in range(i, N):
                    t2 = t.slice(i, j)
                    t3 = t.slice(j, N - 1)
                    if (
                        EmptyInventory.recognize(t2) is not None
                        and RotateTowardsObject.recognize(t3) is not None
                    ):
                        return {
                            "obj": _get_same_object(
                                t.first_state, t.last_state.carrying
                            )
                        }
        return None


class MoveObject(BaseSkill):

    def __init__(self, obj: WorldObj = None, pos: Tuple[int, int] = None):
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
        if t.n_actions == 0 or t.last_action != Actions.drop:
            return None
        # check if t = GetObject(obj) + DropAt(pos)
        N = t.n_states
        for i in range(N):
            t1 = t.slice(0, i)
            ret1 = GetObject.recognize(t1)
            if ret1 is not None:
                t2 = t.slice(i, N - 1)
                ret2 = DropAt.recognize(t2)
                if ret2 is None:
                    continue
                return {
                    "obj": _get_same_object(t.first_state, ret1["obj"]),
                    "pos": ret2["pos"],
                }
        return None


class GoDirNSteps(BaseSkill):

    def __init__(self, dir: int = None, n: int = None):
        self.dir = dir
        self.n = n

    def __call__(self, env: MindGridEnv):
        # rotate to dir
        t = RotateTowardsDirection(self.dir)(env)
        # move forward n steps
        t = t.merge(execute(env, [env.actions.forward] * self.n))
        return t

    def recognize(t):
        N = t.n_states
        for i in range(N):
            t1 = t.slice(0, i)
            ret = RotateTowardsDirection.recognize(t1)
            if ret is not None:
                t2 = t.slice(i, N - 1)
                # all actions in t2 must be forward
                if t2.actions.count(Actions.forward) == t2.n_actions:
                    return {"dir": ret["dir"], "n": t2.n_actions}
        return None


class Unblock(BaseSkill):

    def __init__(self, obj: DoorWithDirection | Bridge = None):
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
            return execute(env, [])

        # otherwise, find a free cell and move blocking object there
        drop_pos = _find_free_cell(env, block_obj.cur_pos)

        if drop_pos is None:
            return NullTrajectory()

        return MoveObject(block_obj, drop_pos)(env)

    def recognize(t):

        def get_blocking_object(s, o):
            adjacent_cells = get_adjacent_cells(o.cur_pos)
            for oo in s.objects:
                if oo.cur_pos in s.outer_cells and oo.cur_pos in adjacent_cells:
                    return oo
            return None

        # must look like a MoveObject
        if MoveObject.recognize(t) is None:
            return None

        # find a door or bridge that is initially blocked but is not at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in last_s.objects:
            if o.type in ["door", "bridge"]:
                blocking_obj = get_blocking_object(first_s, o)
                if blocking_obj is not None and get_blocking_object(last_s, o) is None:
                    return {"obj": _get_same_object(t.first_state, o)}
        return None


class OpenDoor(BaseSkill):
    """Open a door"""

    def __init__(self, door: DoorWithDirection = None):
        self.door = door

    def __call__(self, env: MindGridEnv):
        door = self.door

        # if door is already open, do nothing
        if door.is_open:
            return execute(env, [])

        # unblock door
        t = Unblock(door)(env)

        # if door is locked, get key
        if door.is_locked:
            # if there is no key, can't open
            if not env.keys:
                return NullTrajectory()
            t = t.merge(GetObject(env.keys[0])(env))

        # go to door
        t = t.merge(GoToAdjacentObject(door)(env))
        # toggle door
        t = t.merge(execute(env, [env.actions.toggle]))
        return t

    def recognize(t):
        if t.n_actions == 0:
            for o in t.first_state.objects:
                if o.type == "door" and o.is_open:
                    return {"obj": o}
            return None
        # last action must be toggle
        if t.last_action != Actions.toggle:
            return None
        # find a door that is not open initially but is open at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in first_s.objects:
            if o.type == "door" and not o.is_open:
                for oo in last_s.objects:
                    if oo.init_pos == o.init_pos and oo.is_open:
                        return {"door": _get_same_object(t.first_state, o)}
        return None


class FixBridge(BaseSkill):
    """Fix a bridge"""

    def __init__(self, bridge: Bridge = None):
        self.bridge = bridge

    def __call__(self, env: MindGridEnv):
        bridge = self.bridge

        # if bridge is intact, do nothing
        if bridge.is_intact:
            return execute(env, [])

        # if there is no tool, can't fix
        if not env.tools:
            return NullTrajectory()

        # unblock bridge
        t = Unblock(bridge)(env)
        # get tool
        t = t.merge(GetObject(env.tools[0])(env))
        # go to bridge
        t = t.merge(GoToAdjacentObject(bridge)(env))
        # toggle bridge
        t = t.merge(execute(env, [env.actions.toggle]))
        return t

    def recognize(t):
        if t.n_actions == 0:
            for o in t.first_state.objects:
                if o.type == "bridge" and o.is_intact:
                    return {"obj": o}
            return None
        # last action must be toggle
        if t.last_action != Actions.toggle:
            return None
        # find a bridge that is not intact initially but is intact at the end
        first_s = t.first_state
        last_s = t.last_state
        for o in first_s.objects:
            if o.type == "bridge" and not o.is_intact:
                for oo in last_s.objects:
                    if (
                        oo.type == "bridge"
                        and oo.init_pos == o.init_pos
                        and oo.is_intact
                    ):
                        return {"bridge": _get_same_object(t.first_state, o)}
        return None


class Skills(CustomEnum):

    primitive = Primitive
    goto = GoTo
    rotate_towards_object = RotateTowardsObject
    rotate_towards_direction = RotateTowardsDirection
    goto_adjacent_object = GoToAdjacentObject
    goto_adjacent_position = GoToAdjacentPosition
    drop_at = DropAt
    empty_inventory = EmptyInventory
    get_object = GetObject
    move_object = MoveObject
    go_dir_n_steps = GoDirNSteps
    unblock = Unblock
    open_box = OpenBox
    open_door = OpenDoor
    fix_bridge = FixBridge
