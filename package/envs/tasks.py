from package.infrastructure.env_constants import *
from package.infrastructure.llm_constants import *
from package.enums import Task

from minigrid.core.mission import MissionSpace

from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self):
        if self.is_single_target:
            self.target_obj = None
            self.target_obj_pos = None
        else:
            self.target_objs = []
            self.target_objs_pos = []
    
    @abstractmethod
    def set_mission(self):
        pass


class GoToTask(BaseTask):
    def set_mission(self):
        self.mission = f"get to the {OBJ_NAME_MAPPING[type(self.target_obj)]}"

    # @staticmethod
    def _gen_mission(object: str):
        return f"get to the {object}"
    
    mission_space = MissionSpace(
        mission_func = _gen_mission,
        ordered_placeholders = [PLAYABLE_OBJS]
    )


class PickUpTask(BaseTask):
    def set_mission(self):
        self.mission = f"pick up the {OBJ_NAME_MAPPING[type(self.target_obj)]}"

    # @staticmethod
    def _gen_mission(object: str):
        return f"pick up the {object}"
    
    mission_space = MissionSpace(
        mission_func = _gen_mission,
        ordered_placeholders = [TANGIBLE_OBJS]
    )


class PutNextTask(BaseTask):
    def set_mission(self):
        self.mission = f"put the {OBJ_NAME_MAPPING[type(self.target_objs[0])]} next to the {OBJ_NAME_MAPPING[type(self.target_objs[1])]}"
    
    # @staticmethod
    def _gen_mission(object1: str, object2: str):
        return f"put the {object1} next to the {object2}"
    
    mission_space = MissionSpace(
        mission_func = _gen_mission,
        ordered_placeholders = [TANGIBLE_OBJS, TANGIBLE_OBJS]
    )


class CollectTask(BaseTask):
    def set_mission(self):
        self.mission = f"put all {OBJ_PLURAL_MAPPING[type(self.target_objs[0])]} next to each other"

    # @staticmethod
    def _gen_mission(object: str):
        return f"put all {object} next to each other"
    
    mission_space = MissionSpace(
        mission_func = _gen_mission,
        ordered_placeholders = [TANGIBLE_OBJS]
    )
    

class ClusterTask(BaseTask):
    def set_mission(self):
        if self.env_seed % 2 == 1:
            self.mission = f"group all {self.target_objs[0].color} objects by type"
        else:
            self.mission = f"group all {OBJ_PLURAL_MAPPING[type(self.target_objs[0][0])]} by color"

    # @staticmethod
    def _gen_mission(color: str, object: str):  # FIXME: make cluster by type and cluster by color different tasks
        # if self.env_seed % 2 == 1:
            # return f"group all {color} objects by type"
        return f"group all {object}{'es' if object == 'box' else 's'} by color"
    
    mission_space = MissionSpace(
        mission_func = _gen_mission,
        ordered_placeholders = [COLOR_NAMES, TANGIBLE_OBJS]
    )
