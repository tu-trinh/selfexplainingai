from package.constants import *
from package.utils import *
from package.enums import *
from package.skills import *

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR

import numpy as np
import random


class PragmaticEnv(MiniGridEnv):
    def __init__(self,
                 env_type: EnvType,
                 level: Level,
                 mission_space: MissionSpace,
                 max_steps: int = None,
                 see_through_walls: bool = False,
                 **kwargs):
        np.random.seed(self.env_seed)
        random.seed(self.env_seed)
        self.env_type = env_type
        self.level = level

        # Super init
        if Variant.ROOM_SIZE not in self.variants:
            self.room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        else:
            self.room_size = random.choice(list(set(range(MIN_ROOM_SIZE, MAX_ROOM_SIZE)) - set([self.disallowed[Variant.ROOM_SIZE]])))
        if max_steps is None:
            self.max_steps = 4 * self.room_size ** 2
        else:
            self.max_steps = max_steps
        self.allowable_skills = None
        super().__init__(mission_space = mission_space, grid_size = self.room_size, max_steps = self.max_steps,
                         see_through_walls = see_through_walls, render_mode = "human", agent_view_size = AGENT_VIEW_SIZE, **kwargs)
        
        # Generate random environment
        self.env_id = f"{self.env_type}-{self.level}-{self.env_seed}"
        self.objs = []  # List of (object, (x, y))
        self.walls = []  # List of (wall, x, y)
        self.doors = []  # List of (door, (x, y))
        self.keys = []  # List of (key, x, y)
        self.agent_start_pos = None
        self.agent_start_dir = None
    
    
    def _gen_multiple_rooms(self):
        def helper(self, xlb, xub, ylb, yub):
            pass
        # TODO: figure out this recursiveness
    
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Set walls
        self.grid.wall_rect(0, 0, width, height)
        for i in range(len(self.walls)):
            wall, wall_pos = self.walls[i]
            self.grid.set(wall_pos[0], wall_pos[1], wall)
        wall_positions = [wall[1] for wall in self.walls]
        # Set objects
        for i in range(len(self.objs)):
            obj, obj_pos = self.objs[i]
            assert obj_pos not in wall_positions, "Object cannot be in a wall"
            self.grid.set(obj_pos[0], obj_pos[1], obj)
        # Set doors
        for i in range(len(self.doors)):
            door, door_pos = self.doors[i]
            assert door_pos in wall_positions, "Door can only be for going through walls"
            self.grid.set(door_pos[0], door_pos[1], door)
        # Set keys
        for i in range(len(self.keys)):
            key, key_pos = self.keys[i]
            assert key_pos not in wall_positions, "Key cannot be inside a wall"
            self.grid.set(key_pos[0], key_pos[1], key)
        # Place agent
        if self.agent_start_pos not in wall_positions:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    

    def set_allowable_skills(self):
        skills = {}  # maps skill name to function
        can_pickup_and_drop = False
        can_toggle = False

        # Directional skills
        for d in ["forward", "left", "right", "backward"]:
            for i in range(1, self.room_size - 2):
                skills[f"move_{d}_{i}_steps"] = move_DIRECTION_N_steps_hof(d, i)

        # Object-based skills
        fully_obs_copy = FullyObsWrapper(self)
        obs, _ = fully_obs_copy.reset()
        obs = obs["image"]
        for r in range(len(obs)):
            for c in range(len(obs[0])):
                cell = obs[r][c]
                obj_idx, color_idx, _ = cell[0], cell[1], cell[2]
                obj, color = IDX_TO_OBJECT[obj_idx], IDX_TO_COLOR[color_idx]
                if obj in ["wall", "lava"]:
                    skills.setdefault(f"go_to_{obj}", go_to_COLOR_OBJECT_hof(color, obj))
                elif obj in ["door", "box"]:
                    can_toggle = True
                    skills.setdefault(f"go_to_{color}_{obj}", go_to_COLOR_OBJECT_hof(color, obj))
                    skills.setdefault(f"open_{color}_{obj}", open_COLOR_OBJECT_hof(color, obj))
                    if obj == "door":
                        skills.setdefault(f"close_{color}_{obj}", close_COLOR_door_hof(color))
                        skills.setdefault(f"unlock_{color}_{obj}", unlock_COLOR_door_hof(color))
                    elif obj == "box":
                        can_pickup_and_drop = True
                        skills.setdefault(f"pickup_{color}_{obj}", pickup_COLOR_OBJECT_hof(color, obj))
                        skills.setdefault(f"put_down_{color}_{obj}", put_down_COLOR_OBJECT_hof(color, obj))
                elif obj in ["goal", "ball", "key"]:
                    skills.setdefault(f"go_to_{color}_{obj}", go_to_COLOR_OBJECT_hof(color, obj))
                    if obj != "goal":
                        can_pickup_and_drop = True
                        skills.setdefault(f"pickup_{color}_{obj}", pickup_COLOR_OBJECT_hof(color, obj))
                        skills.setdefault(f"put_down_{color}_{obj}", put_down_COLOR_OBJECT_hof(color, obj))
        
        # Primitive Minigrid skills
        skills["turn_left"] = turn_left
        skills["turn_right"] = turn_right
        skills["forward"] = forward
        if can_pickup_and_drop:
            skills["pickup"] = pickup
            skills["drop"] = drop
        if can_toggle:
            skills["toggle"] = toggle
        
        self.allowable_skills = skills
