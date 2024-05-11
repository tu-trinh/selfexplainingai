from __future__ import annotations

from abc import ABC, abstractmethod

from mindgrid.infrastructure.env_utils import bfs
from mindgrid.infrastructure.basic_utils import get_adjacent_cells
from mindgrid.infrastructure.trajectory import NullTrajectory
import mindgrid.skills as skill_lib


class BaseSolver(ABC):

    @abstractmethod
    def solve_with_optimal_skills(self):
        raise NotImplementedError


class RoomDoorKeySolver(BaseSolver):

    def solve_with_optimal_skills(self) -> Trajectory:

        if (
            bfs(
                self.gen_simple_2d_map(),
                self.agent_dir,
                self.agent_pos,
                get_adjacent_cells(self.targets[0].cur_pos),
            )
            is None
        ):
            if self.doors:
                # check if there is an open door
                open_door = None
                for d in self.doors:
                    if d.is_open:
                        open_door = d
                        break
                if open_door is None:
                    # choose a door and open it
                    open_door = self.random.choice(self.doors)
                    t = skill_lib.OpenDoor(open_door)(self)
                else:
                    t = skill_lib.execute(self, [])
                t = t.merge(skill_lib.Unblock(open_door)(self))
            else:
                # no doors and can't reach goal -> no solution
                return NullTrajectory()
        else:
            t = skill_lib.execute(self, [])

        t = t.merge(skill_lib.GetObject(self.targets[0])(self))

        return t


class TreasureIslandSolver(BaseSolver):

    def solve_with_optimal_skills(self) -> Trajectory:

        if (
            bfs(
                self.gen_simple_2d_map(),
                self.agent_dir,
                self.agent_pos,
                get_adjacent_cells(self.targets[0].cur_pos),
            )
            is None
        ):
            fireproof_shoes = None
            for o in self.objects:
                if o.type == "fireproof_shoes":
                    fireproof_shoes = o
                    break
            # if fireproof shoes are present, grab them
            if fireproof_shoes is not None:
                t = skill_lib.GetObject(fireproof_shoes)(self)
            elif self.bridges:
                # find an intact bridge
                intact_bridge = None
                for b in self.bridges:
                    if b.is_intact:
                        intact_bridge = b
                        break
                if intact_bridge is None:
                    # choose a bridge and fix it
                    intact_bridge = self.random.choice(self.bridges)
                    t = skill_lib.FixBridge(intact_bridge)(self)
                else:
                    t = skill_lib.execute(self, [])
                t = t.merge(skill_lib.Unblock(intact_bridge)(self))
            else:
                # no bridges and shoes and can't reach goal -> no solution
                return NullTrajectory()
        else:
            t = skill_lib.execute(self, [])

        t = t.merge(skill_lib.GetObject(self.targets[0])(self))

        return t

