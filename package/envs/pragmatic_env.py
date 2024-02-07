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

        # Special case of orientation variant
        if Variant.ORIENTATION in self.variants:
            self._gen_rotated_room(self.disallowed[Variant.ORIENTATION], random.choice([90, 180, 270]))
    
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, 0, False, False, info

    
    def _gen_path_to_target(self):
        total_path = set()
        if self.env_type == EnvType.GOTO or self.env_type == EnvType.PICKUP:
            targets = [self.target_obj_pos]
        else:
            targets = self.target_objs_pos
        for target_pos in targets:
            path = [tuple(self.agent_start_pos)]
            pos = self.agent_start_pos
            direction = self.agent_start_dir
            max_turns = random.randint(1, self.room_size - 4)
            num_turns = 0
            reached_object = False
            while not reached_object and num_turns < max_turns:
                if direction == 0:  # right
                    steps_ub = self.room_size - 2 - pos[0]
                    delta = (1, 0)
                elif direction == 1:  # down
                    steps_ub = self.room_size - 2 - pos[1]
                    delta = (0, 1)
                elif direction == 2:  # left
                    steps_ub = pos[0] - 1
                    delta = (-1, 0)
                elif direction == 3:  # up
                    steps_ub = pos[1] - 1
                    delta = (0, -1)
                if steps_ub <= 1:
                    direction = random.randint(0, 4)
                    continue
                num_steps = random.randint(1, steps_ub)
                for _ in range(num_steps):
                    new_pos = (pos[0] + delta[0], pos[1] + delta[1])
                    path.append(new_pos)
                    pos = new_pos
                    if pos == target_pos:
                        reached_object = True
                        break
                direction = min(3, direction + 1) if np.random.random() > 0.5 else max(0, direction - 1)
                num_turns += 1
            if pos != target_pos:
                if pos[0] < target_pos[0]:
                    horizontal_step = 1
                elif pos[0] > target_pos[0]:
                    horizontal_step = -1
                else:
                    horizontal_step = None
                if pos[1] < target_pos[1]:
                    vertical_step = 1
                elif pos[1] > target_pos[1]:
                    vertical_step = -1
                else:
                    vertical_step = None
                if horizontal_step:
                    for x in range(pos[0] + horizontal_step, target_pos[0] + horizontal_step, horizontal_step):
                        new_pos = (x, pos[1])
                        path.append(new_pos)
                        pos = new_pos
                if vertical_step:
                    for y in range(pos[1] + vertical_step, target_pos[1] + vertical_step, vertical_step):
                        new_pos = (pos[0], y)
                        path.append(new_pos)
                        pos = new_pos
            total_path.update(set(path))
        return total_path
    
    
    def _get_cells_in_partition(self, start_x, end_x, start_y, end_y):
        return [(x, y) for x in range(start_x, end_x + 1) for y in range(start_y, end_y + 1)]
    
    
    def _generate_walls_for_partition(self, start_x, end_x, start_y, end_y, min_subroom_size = 2):
        walls = []
        door = None  # Initialize door as None; it may remain None based on randomness
        # Determine if we're splitting vertically or horizontally based on the larger dimension
        split_vertically = (end_x - start_x) > (end_y - start_y)
        if split_vertically:
            # Ensure there's enough space for a subroom on either side of the wall
            if (end_x - start_x) < 2 * min_subroom_size:
                return walls, door  # Not enough space to split this partition further
            wall_x = random.randint(start_x + min_subroom_size, end_x - min_subroom_size)
            slit_y = random.randint(start_y, end_y)
            for y in range(start_y, end_y + 1):
                if y != slit_y:  # Skip the slit position
                    walls.append((wall_x, y))
                elif random.random() < 0.5:  # Random chance to turn slit into a door
                    door = (wall_x, y)
        else:
            if (end_y - start_y) < 2 * min_subroom_size:
                return walls, door
            wall_y = random.randint(start_y + min_subroom_size, end_y - min_subroom_size)
            slit_x = random.randint(start_x, end_x)
            for x in range(start_x, end_x + 1):
                if x != slit_x:  # Skip the slit position
                    walls.append((x, wall_y))
                elif random.random() < 0.5:  # Random chance to turn slit into a door
                    door = (x, wall_y)
        return walls, door
    
    
    def _gen_multiple_rooms(self):
        walls = []
        doors = []
        partitions = [(1, self.room_size - 2, 1, self.room_size - 2)]  # Initial partition covering the whole room
        partition_cells = []
        while len(partitions) < self.num_rooms:
            # Randomly select a partition to split
            partition_to_split = random.choice(partitions)
            partitions.remove(partition_to_split)
            start_x, end_x, start_y, end_y = partition_to_split    
            # Generate walls within the selected partition
            new_walls, door = self._generate_walls_for_partition(start_x, end_x, start_y, end_y)
            if not new_walls and not door:
                partitions.append(partition_to_split)  # Revert if no walls or door were added
                continue
            if door:
                doors.append(door)
            # Determine new partitions created by the wall
            if new_walls[0][0] == new_walls[-1][0]:  # Vertical wall
                left_partition = (start_x, new_walls[0][0] - 1, start_y, end_y)
                right_partition = (new_walls[0][0] + 1, end_x, start_y, end_y)
                partitions.extend([left_partition, right_partition])
            else:  # Horizontal wall
                top_partition = (start_x, end_x, start_y, new_walls[0][1] - 1)
                bottom_partition = (start_x, end_x, new_walls[0][1] + 1, end_y)
                partitions.extend([top_partition, bottom_partition])
            walls.extend(new_walls)
        # Generate cell lists for each partition
        for partition in partitions:
            start_x, end_x, start_y, end_y = partition
            cells = self._get_cells_in_partition(start_x, end_x, start_y, end_y)
            partition_cells.append(set([cell for cell in cells if cell not in walls or cell in doors]))
        return walls, doors, partition_cells
    
    
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
    

    def _gen_rotated_room(self, ref_env, degrees):
        rotation_deltas = {
            90: lambda x, y: (ref_env.room_size - 1 - y, x),
            180: lambda x, y: (ref_env.room_size - 1 - x, ref_env.room_size - 1 - y),
            270: lambda x, y: (y, ref_env.room_size - 1 - x)
        }
        delta_func = rotation_deltas[degrees]
        for obj, pos in ref_env.objs:
            self.objs.append((obj, delta_func(*pos)))
        for wall, pos in ref_env.walls:
            self.walls.append((wall, delta_func(*pos)))
        for door, pos in ref_env.doors:
            self.doors.append((door, delta_func(*pos)))
        for key, pos in ref_env.keys:
            self.keys.append((key, delta_func(*pos)))
        self.agent_start_pos = delta_func(*ref_env.agent_start_pos)
        self.agent_start_dir = ref_env.agent_start_dir  # dir does not rotate otherwise it's just the same room
        if self.env_type in [EnvType.GOTO, EnvType.PICKUP]:
            self.target_obj = ref_env.target_obj
            self.target_obj_pos = delta_func(*ref_env.target_obj_pos)
        else:
            self.target_objs = ref_env.target_objs
            self.target_objs_pos = [delta_func(*obj_pos) for obj_pos in ref_env.target_objs_pos]
        if self.level == Level.MULT_ROOMS:
            self.num_rooms = ref_env.num_rooms
    
    
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
