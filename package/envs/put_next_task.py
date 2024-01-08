from minigrid.core.mission import MissionSpace
from constants import *
from utils import *
from envs.enums import *
from envs.multi_target_env import MultiTargetEnv


class PutTask(MultiTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, TANGIBLE_OBJS, OBJECT_COLOR_NAMES, TANGIBLE_OBJS])
        super().__init__(EnvType.PUT, env_seed, level, mission_space, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
    
    @staticmethod
    def _gen_mission(color1: str, object1: str, color2: str, object2: str):
        return f"put the {color1} {object1} next to the {color2} {object2}"
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        u, v = self.dir_vec
        px, py = self.agent_pos[0] + u, self.agent_pos[1] + v
        tx, ty = self.target_objs_pos[1]
        if action == self.actions.drop and self.grid.get(px, py) == self.target_objs[0]:
            if abs(px - tx) <= 1 and abs(py - ty) <= 1:
                reward = self._reward()
                terminated = True
        return obs, reward, terminated, truncated, info