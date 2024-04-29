from package.infrastructure.env_constants import *
from package.infrastructure.obj_constants import *
from package.infrastructure.llm_constants import *
from package.enums import Task

from minigrid.core.world_object import WorldObj, Door, Key, Goal, Wall, Lava, Box
from minigrid.core.mission import MissionSpace

import numpy as np
from typing import List, Tuple, Type
from abc import ABC, abstractmethod


class BaseTask(ABC):

    def _init_task(self):
        self.target_color =  self.random.choice(self.allowed_object_colors)
        self._make_target_objects()
        self._set_mission()

    @abstractmethod
    def _set_mission(self):
        pass

    @abstractmethod
    def _gen_mission(*args, **kwargs):
        pass

    @abstractmethod
    def _make_target_objects(self) -> None:
        pass


class PickUpTask(BaseTask):
    def _set_mission(self):
        self.mission = f"pick up all {self.target_objects[0].color} {OBJ_NAME_MAPPING[type(self.target_objects[0])]}s"

    def _gen_mission(object: str):
        return f"pick up the {object}"

    mission_space = MissionSpace(
        mission_func=_gen_mission, ordered_placeholders=[TANGIBLE_OBJS]
    )

    def _make_target_objects(self) -> None:
        self.target_objects = [Ball(color=self.target_color)]
