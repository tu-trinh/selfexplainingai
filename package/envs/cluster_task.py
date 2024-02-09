from package.constants import *
from package.utils import *
from package.enums import *
from package.envs.multi_target_env import MultiTargetEnv

from minigrid.core.world_object import WorldObj
from minigrid.core.mission import MissionSpace

from typing import Dict, Any


class ClusterTask(MultiTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 target_objs: List[WorldObj] = None,
                 variants: List[Variant] = None,
                 disallowed: Dict[Variant, Any] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        self.env_seed = env_seed
        self.variants = variants if variants is not None else []
        self.disallowed = disallowed if disallowed is not None else {}

        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, TANGIBLE_OBJS])  # FIXME: how does this work with the gen mission below lol
        super().__init__(EnvType.CLUSTER,
                         level,
                         mission_space,
                         target_objs = target_objs,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
    
    @staticmethod
    def _gen_mission(color: str, object: str):
        if np.random.random() < 0.5:  # TODO: make sure this static method isn't actually needed => seed isn't needed
            return f"group all {color} objects by type"
        return f"group all {object}{'es' if object == 'box' else 's'} by color"
