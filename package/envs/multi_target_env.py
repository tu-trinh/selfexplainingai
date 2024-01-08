from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
from minigrid.core.mission import MissionSpace
from constants import *
import numpy as np
import random
from utils import *
from envs.pragmatic_env import PragmaticEnv
from envs.enums import *


class MultiTargetEnv(PragmaticEnv):
    def __init__(self,
                 env_type: EnvType,
                 env_seed: int,
                 level: Level,
                 mission_space: MissionSpace,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        assert env_type not in [EnvType.GOTO, EnvType.PICKUP], "Env type can't be Goto or Pickup"
        assert level in [Level.EMPTY]

        super().__init__(env_type, env_seed, level, mission_space, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
        
        # Generate random environment
        self.target_objs = []
        self.target_objs_pos = []
        all_possible_pos = set([(x, y) for x in range(1, self.room_size - 1) for y in range(1, self.room_size - 1)])

        if level in [Level.EMPTY]:
            self.agent_start_pos = random.choice(list(all_possible_pos))
            all_possible_pos -= set([self.agent_start_pos])
            if self.env_type == EnvType.PUT:
                all_possible_pos = list(all_possible_pos)
                a, b = (0, 0), (0, 0)
                while abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
                    positions = np.random.choice(len(all_possible_pos), 2, replace = False)
                    a = all_possible_pos[positions[0]]
                    b = all_possible_pos[positions[1]]
                self.target_objs_pos = [a, b]
                all_possible_pos = set(all_possible_pos)
                all_possible_pos -= set(self.target_objs_pos)
            elif self.env_type == EnvType.COLLECT:
                num_collectibles = random.choice(range(2, 5))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(env_seed)
            self.objs = list(zip(self.target_objs, self.target_objs_pos))

            # if level == Level.DIST:
            #     num_distractors = np.random.choice(range(1, self.room_size - 3))
            #     for _ in range(num_distractors):
            #         dist_obj = self.target_obj
            #         while type(dist_obj) == type(self.target_obj) and dist_obj.color == self.target_obj.color:
            #             dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
            #         dist_obj_pos = random.choice(list(all_possible_pos))
            #         all_possible_pos -= set([dist_obj_pos])
            #         self.objs.append((dist_obj, dist_obj_pos))
        
        self.agent_start_dir = np.random.randint(0, 4)
        self._gen_grid(self.room_size, self.room_size)
            
        # Set mission
        if self.env_type == EnvType.PUT:
            self.mission = f"put the {self.target_objs[0].color} {OBJ_NAME_MAPPING[type(self.target_objs[0])]} next to the {self.target_objs[1].color} {OBJ_NAME_MAPPING[type(self.target_objs[1])]}"
        elif self.env_type == EnvType.COLLECT:
            self.mission = f"put all {self.target_objs[0].color} {OBJ_PLURAL_MAPPING[type(self.target_objs[0])]} next to each other"
        
        # Final asserts
        assert self.agent_start_pos is not None, "self.agent_start_pos is None"
        assert self.agent_start_dir is not None, "self.agent_start_dir is None"
        assert len(self.target_objs) != 0, "self.target_objs is empty"
        assert len(self.target_objs_pos) != 0, "self.target_objs_pos is empty"
    
    def _set_target_objs(self, env_seed):
        if self.env_type == EnvType.PUT:
            idx1 = env_seed % len(TANGIBLE_OBJS)
            idx2 = (env_seed + 1) % len(PLAYABLE_OBJS)
            color1 = OBJECT_COLOR_NAMES[env_seed % len(OBJECT_COLOR_NAMES)]
            color2 = OBJECT_COLOR_NAMES[(env_seed + 1) % len(OBJECT_COLOR_NAMES)]
            pickup_obj = TANGIBLE_OBJS[idx1](color = color1)
            target_obj = PLAYABLE_OBJS[idx2]
            if target_obj == Goal:
                target_obj = Goal()
                target_obj.color = color2
            else:
                target_obj = target_obj(color = color2)
            self.target_objs = [pickup_obj, target_obj]
        elif self.env_type == EnvType.COLLECT:
            collectible_obj = TANGIBLE_OBJS[env_seed % len(TANGIBLE_OBJS)]
            color = OBJECT_COLOR_NAMES[env_seed % len(OBJECT_COLOR_NAMES)]
            for _ in range(len(self.target_objs_pos)):
                self.target_objs.append(collectible_obj(color = color))