from minigrid.core.mission import MissionSpace
from constants import *
from utils import *
from envs.single_target_env import SingleTargetEnv
from envs.enums import *
from typing import Tuple, List, Dict, Any


class PickupTask(SingleTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 target_obj: Tuple[PLAYABLE_OBJS] = None,
                 variants: List[Variant] = None,
                 disallowed: Dict[Variant, Any] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        self.env_seed = env_seed
        self.variants = variants if variants is not None else []
        self.disallowed = disallowed if disallowed is not None else {}

        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, TANGIBLE_OBJS])
        super().__init__(EnvType.PICKUP,
                         level,
                         mission_space,
                         target_obj = target_obj,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
    
    @staticmethod
    def _gen_mission(color: str, object: str):
        return f"pick up the {color} {object}"
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.target_obj:
                reward = self._reward()
                terminated = True
        return obs, reward, terminated, truncated, info