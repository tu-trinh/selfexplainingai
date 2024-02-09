from package.constants import *
from package.utils import *
from package.enums import *
from package.envs.multi_target_env import MultiTargetEnv

from minigrid.core.world_object import WorldObj
from minigrid.core.mission import MissionSpace

from typing import Dict, Any


class CollectTask(MultiTargetEnv):
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
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, TANGIBLE_OBJS])
        super().__init__(EnvType.COLLECT,
                         level,
                         mission_space,
                         target_objs = target_objs,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
    
    @staticmethod
    def _gen_mission(color: str, object: str):
        return f"put all {color} {object} next to each other"
    
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     if action == self.actions.drop:
    #         obj_positions = []
    #         for i in range(len(self.target_objs)):
    #             if self.target_objs[i].cur_pos is None:
    #                 obj_positions.append(self.target_objs_pos[i])
    #             else:
    #                 obj_positions.append(tuple(self.target_objs[i].cur_pos))
    #         if self._all_adjacent(obj_positions):
    #             reward = self._reward()
    #             terminated = True
    #     return obs, reward, terminated, truncated, info

    # def _all_adjacent(self, positions):
    #     def find_path(current, remaining):
    #         if not remaining:
    #             return True
    #         for next_pos in remaining:
    #             if abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1]) == 1:
    #                 if find_path(next_pos, [pos for pos in remaining if pos != next_pos]):
    #                     return True
    #         return False
    #     start_pos = positions[0]
    #     remaining_positions = positions[1:]
    #     return find_path(start_pos, remaining_positions)
