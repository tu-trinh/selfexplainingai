from package.constants import *
from package.utils import *
from package.envs.single_target_env import SingleTargetEnv
from package.enums import *

from minigrid.core.world_object import Goal, WorldObj
from minigrid.core.mission import MissionSpace

from typing import List, Tuple, Dict, Any


class GotoTask(SingleTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 target_obj: WorldObj = None,
                 variants: List[Variant] = None,
                 disallowed: Dict[Variant, Any] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        self.env_seed = env_seed
        self.variants = variants if variants is not None else []
        self.disallowed = disallowed if disallowed is not None else {}
        
        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, PLAYABLE_OBJS])
        super().__init__(EnvType.GOTO,
                         level,
                         mission_space,
                         target_obj = target_obj,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
    
    
    @staticmethod
    def _gen_mission(color: str, object: str):
        return f"get to the {color} {object}"
    

    # def step(self, action):
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     ax, ay = self.agent_pos
    #     tx, ty = self.target_obj_pos
    #     if type(self.target_obj) == Goal:
    #         if ax == tx and ay == ty:
    #             reward = self._reward()
    #             terminated = True
    #     else:
    #         if manhattan_distance((ax, ay), (tx, ty)) == 1:
    #             target_obj_dir = (tx - ax == 1, ty - ay == 1, ax - tx == 1, ay - ty == 1).index(True)
    #             if self.agent_dir == target_obj_dir:
    #                 reward = self._reward()
    #                 terminated = True
    #     return obs, reward, terminated, truncated, info
