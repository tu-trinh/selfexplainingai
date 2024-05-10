from __future__ import annotations

from abc import ABC, abstractmethod

from mindgrid.infrastructure.env_utils import bfs
from mindgrid.infrastructure.basic_utils import get_adjacent_cells
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
                get_adjacent_cells(self.target_objects[0].cur_pos),
            )
            is None
        ):
            if self.doors:
                # choose a door and open it
                target_door = self.random.choice(self.doors)
                t = skill_lib.OpenDoor(target_door)(self)
                if t is None:
                    return None
                t = t.merge(skill_lib.Unblock(target_door)(self))
            else:
                # no doors and can't reach goal -> no solution
                return None
        else:
            t = skill_lib.execute(self, [])

        # check if target object is in a box
        box_with_target_object = None
        for o in self.objects:
            if o.type == "box" and o.contains == self.target_objects[0]:
                box_with_target_object = o
                break

        # open box containing target object
        if box_with_target_object is not None:
            t = t.merge(skill_lib.OpenBox(box_with_target_object)(self))

        # get target object
        t = t.merge(skill_lib.GoToAdjacentObject(self.target_objects[0])(self))
        t = t.merge(skill_lib.GetObject(self.target_objects[0])(self))

        return t


class TreasureIslandSolver(BaseSolver):

    def solve_with_optimal_skills(self) -> Trajectory:

        if (
            bfs(
                self.gen_simple_2d_map(),
                self.agent_dir,
                self.agent_pos,
                get_adjacent_cells(self.target_objects[0].cur_pos),
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
                # choose a bridge and fix it
                target_bridge = self.random.choice(self.bridges)
                t = skill_lib.FixBridge(target_bridge)(self)
                if t is None:
                    return None
                t = t.merge(skill_lib.Unblock(target_bridge)(self))
            else:
                # no bridges and shoes and can't reach goal -> no solution
                return None
        else:
            t = skill_lib.execute(self, [])


        """
        # check if target object is in a box
        box_with_target_object = None
        for o in self.objects:
            if o.type == "box" and o.contains == self.target_objects[0]:
                box_with_target_object = o
                break

        # open box containing target object
        if box_with_target_object is not None:
            t = t.merge(skill_lib.OpenBox(box_with_target_object)(self))

        # get target object
        t = t.merge(skill_lib.GoToAdjacentObject(self.target_objects[0])(self))
        t = t.merge(skill_lib.GetObject(self.target_objects[0])(self))
        """

        t = t.merge(skill_lib.GetObject(self.target_objects[0])(self))

        return t

