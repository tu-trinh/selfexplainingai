from abc import ABC, abstractmethod

from minigrid.core.mission import MissionSpace

from mindgrid.infrastructure.basic_utils import CustomEnum
from mindgrid.infrastructure.env_constants import *
from mindgrid.infrastructure.obj_constants import *


class BaseTask(ABC):

    def _init_task(self):
        self.target_color = self.random.choice(sorted(self.allowed_object_colors))
        self._set_mission()

    @abstractmethod
    def _set_mission(self):
        pass

    @abstractmethod
    def _gen_mission(*args, **kwargs):
        pass


class PickUpTask(BaseTask):

    def _set_mission(self):
        self.mission = f"pick up the {self.target_color} {OBJ_NAME_MAPPING[self.target_cls]}"

    def _gen_mission(object: str):
        return f"pick up the {object}"

    mission_space = MissionSpace(
        mission_func=_gen_mission, ordered_placeholders=[TANGIBLE_OBJS]
    )

    @property
    def target_cls(self):
        return Ball


class Tasks(CustomEnum):

    pickup = PickUpTask
    #TODO: add more tasks here
