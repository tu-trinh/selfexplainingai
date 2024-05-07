from abc import ABC, abstractmethod

from minigrid.core.mission import MissionSpace

from mindgrid.infrastructure.basic_utils import CustomEnum
from mindgrid.infrastructure.env_constants import *
from mindgrid.infrastructure.obj_constants import *


class BaseTask(ABC):

    def _init_task(self):
        self.target_color = self.random.choice(self.allowed_object_colors)
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


class Task(CustomEnum):

    PICKUP = PickUpTask
