from package.constants import *
from package.utils import *
from package.envs.multi_target_env import MultiTargetEnv
from package.enums import *

from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj

from typing import Dict, Any


class PutNextTask(MultiTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 target_objs: List[WorldObj] = None,
                 variants: List[Variant] = None,
                 disallowed: Dict[Variant, Any] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 render_mode = None,
                 **kwargs):
        self.env_seed = env_seed
        self.variants = variants if variants is not None else []
        self.disallowed = disallowed if disallowed is not None else {}
        self.render_mode = render_mode

        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, TANGIBLE_OBJS, OBJECT_COLOR_NAMES, TANGIBLE_OBJS])
        super().__init__(EnvType.PUT,
                         level,
                         mission_space,
                         target_objs = target_objs,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
    
    @staticmethod
    def _gen_mission(color1: str, object1: str, color2: str, object2: str):
        return f"put the {color1} {object1} next to the {color2} {object2}"
    
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     u, v = self.dir_vec
    #     px, py = self.agent_pos[0] + u, self.agent_pos[1] + v
    #     tx, ty = self.target_objs_pos[1]
    #     if action == self.actions.drop and self.grid.get(px, py) == self.target_objs[0]:
    #         if abs(px - tx) <= 1 and abs(py - ty) <= 1:
    #             reward = self._reward()
    #             terminated = True
    #     return obs, reward, terminated, truncated, info
