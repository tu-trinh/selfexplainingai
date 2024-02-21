from package.constants import *
from package.enums import *
from package.utils import *
from package.skills import *
from package.envs.tasks import *
from package.envs.levels import *

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR
from minigrid.core.world_object import Wall, Box, WorldObj

from typing import Dict, Any
import numpy as np
import random
import warnings


task_class_mapping = {
    Task.GOTO: GoToTask,
    Task.PICKUP: PickUpTask,
    Task.PUT: PutNextTask,
    Task.COLLECT: CollectTask,
    Task.CLUSTER: ClusterTask
}
level_class_mapping = {
    Level.EMPTY: EmptyLevel,
    Level.DEATH: DeathLevel,
    Level.DIST: DistractorsLevel,
    Level.OPEN_DOOR: OpenDoorLevel,
    Level.BLOCKED_DOOR: BlockedDoorLevel,
    Level.UNLOCK_DOOR: UnlockDoorLevel,
    Level.HIDDEN_KEY: HiddenKeyLevel,
    Level.GO_AROUND: GoAroundLevel,
    Level.MULT_ROOMS: MultipleRoomsLevel,
    Level.ROOM_DOOR_KEY: RoomDoorKeyLevel,
    Level.TREASURE_ISLAND: TreasureIslandLevel,
    Level.BOSS: BossLevel
}


class Environment(MiniGridEnv):
    def __init__(self,
                 env_seed: int,
                 task: Task,
                 level: Level,
                 target_obj: WorldObj = None,
                 target_objs: List[WorldObj] = None,
                 disallowed: Dict[Variant, Any] = None,
                 max_steps: int = None,
                 agent_view_size: int = 5,
                 render_mode = None,
                 **kwargs):
        # Instance variables
        self.env_seed = env_seed
        self.task = task
        self.level = level
        self.disallowed = disallowed if disallowed else {}
        self.render_mode = render_mode
        self.agent_view_size = agent_view_size
        self.is_single_target = self.task in [Task.GOTO, Task.PICKUP]
        if self.is_single_target:
            self.target_obj_type = target_obj
        else:
            self.target_obj_types = target_objs

        # Set random seeds
        np.random.seed(self.env_seed)
        random.seed(self.env_seed)

        # Set environment configs
        if Variant.ROOM_SIZE not in self.disallowed:
            self.room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        else:
            self.room_size = random.choice(list(set(range(MIN_ROOM_SIZE, MAX_ROOM_SIZE)) - set([self.disallowed[Variant.ROOM_SIZE]])))
        if max_steps is None:
            self.max_steps = 4 * self.room_size ** 2
        else:
            self.max_steps = max_steps
        self.allowable_skills = None

        # Generate task mission space and super init
        task_class = task_class_mapping[self.task]
        mission_space = task_class.mission_space
        super().__init__(
            mission_space = mission_space,
            grid_size = self.room_size,
            max_steps = self.max_steps,
            see_through_walls = False,
            render_mode = self.render_mode,
            agent_view_size = self.agent_view_size,
            **kwargs
        )

        # Unique environment configs
        self.env_id = f"{self.task}-{self.level}-{self.env_seed}"
        self.objs = []  # List of (object, (x, y))
        self.walls = []  # List of (wall, x, y)
        self.doors = []  # List of (door, (x, y))
        self.keys = []  # List of (key, x, y)
        self.agent_start_pos = None
        self.agent_start_dir = None

        # Special case of orientation variant
        if Variant.ORIENTATION in self.disallowed:
            self._gen_rotated_room(random.choice([90, 180, 270]), ref_env = self.disallowed[Variant.ORIENTATION])
    
    """
    Public Methods
    """
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, 0, terminated, truncated, info
    

    def set_allowable_skills(self):
        skills = {}  # maps skill name to function
        can_pickup_and_drop = False
        can_toggle = False

        # Directional skills
        for d in ["forward", "left", "right", "backward"]:
            for i in range(1, self.room_size - 2):
                skills[f"move_{d}_{i}_steps"] = move_direction_n_steps_hof(d, i)

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
                    skills.setdefault(f"go_to_{obj}", go_to_color_object_hof(color, obj))
                elif obj in ["door", "box"]:
                    can_toggle = True
                    skills.setdefault(f"go_to_{color}_{obj}", go_to_color_object_hof(color, obj))
                    skills.setdefault(f"open_{color}_{obj}", open_color_object_hof(color, obj))
                    if obj == "door":
                        skills.setdefault(f"close_{color}_{obj}", close_color_door_hof(color))
                        skills.setdefault(f"unlock_{color}_{obj}", unlock_color_door_hof(color))
                    elif obj == "box":
                        can_pickup_and_drop = True
                        skills.setdefault(f"pickup_{color}_{obj}", pickup_color_object_hof(color, obj))
                        skills.setdefault(f"put_down_{color}_{obj}", put_down_color_object_hof(color, obj))
                elif obj in ["goal", "ball", "key"]:
                    skills.setdefault(f"go_to_{color}_{obj}", go_to_color_object_hof(color, obj))
                    if obj != "goal":
                        can_pickup_and_drop = True
                        skills.setdefault(f"pickup_{color}_{obj}", pickup_color_object_hof(color, obj))
                        skills.setdefault(f"put_down_{color}_{obj}", put_down_color_object_hof(color, obj))
        
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
    

    def change_room_size(self, new_size: int = None):
        if not new_size:
            new_size = random.choice(list(range(MIN_ROOM_SIZE, self.room_size)) + list(range(self.room_size + 1, MAX_ROOM_SIZE)))
        size_delta = new_size - self.room_size
        debug("old", self.room_size, "new", new_size)
        offset = abs(size_delta) // 2 if size_delta < 0 else size_delta // 2
        new_doors, new_keys, new_walls, new_objs = [], [], [], []

        def calculate_new_position(pos):
            x, y = pos
            if size_delta > 0:
                return (x + offset, y + offset)
            else:
                return (x - offset, y - offset)
        
        def is_within_bounds(pos):
            x, y = pos
            return 1 <= x < new_size - 1 and 1 <= y < new_size - 1
        
        def find_next_available_position(pos):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if is_within_bounds(new_pos):
                        return new_pos
            return None
        
        def calculate_new_wall_assignments(wall_detection):
            ratios = []
            curr_walls = 0
            curr_cleft = 0
            if wall_detection[0]:
                curr_walls += 1
                currently_wall = True
            else:
                curr_cleft += 1
                currently_wall = False
            for i in range(1, len(wall_detection)):
                if currently_wall and wall_detection[i]:  # continuing the wall
                    curr_walls += 1
                elif currently_wall and not wall_detection[i]:  # found a cleft
                    ratios.append((curr_walls / (self.room_size - 2), True))
                    curr_walls = 0
                    curr_cleft += 1
                elif not currently_wall and not wall_detection[i]:  # continuing the cleft
                    curr_cleft += 1
                else:  # found a wall
                    ratios.append((curr_cleft / (self.room_size - 2), False))
                    curr_cleft = 0
                    curr_walls += 1
            if currently_wall:
                ratios.append((curr_walls / (self.room_size - 2), True))
            else:
                ratios.append((curr_cleft / (self.room_size - 2), False))
            new_assignments = []
            for i in range(len(ratios) - 1):
                assignment = (np.ceil(ratios[i][0] * new_size), ratios[i][1])
                new_assignments.append(assignment)
            if len(ratios) > 0:
                new_assignments.append((new_size - 2 - sum([cells for cells, _ in new_assignments]), ratios[-1][1]))
            return new_assignments
        
        def calculate_new_wall_positions(wall_distribution):
            splits = np.where(wall_distribution)[0]
            non_wall_sections = []
            for i in range(len(splits)):
                if i == 0:
                    non_wall_sections.append(wall_distribution[:splits[i]])
                else:
                    non_wall_sections.append(wall_distribution[splits[i - 1] + 1 : splits[i]])
            non_wall_sections.append(wall_distribution[splits[-1] + 1:])
            j = 0
            for i in range(size_delta):
                non_wall_sections[j].append(False)
                j = (j + 1) % len(non_wall_sections)
            total_sections = []
            for i, section in enumerate(non_wall_sections):
                total_sections.append(section)
                if i < len(non_wall_sections) - 1:
                    total_sections.append([True])
            return total_sections
        
        def fill_new_walls(sections, assignments, is_horizontal_walls):
            sections = flatten_list(sections)
            j = 0
            for i in range(new_size - 2):
                if sections[i]:
                    k = 1
                    try:  # might not be any assignments
                        for num_cells, is_wall in assignments[j]:
                            for _ in range(int(num_cells)):
                                if is_wall:
                                    if is_horizontal_walls:
                                        new_walls.append((Wall(), (k, i + 1)))
                                    else:
                                        new_walls.append((Wall(), (i + 1, k)))
                                k += 1
                        j += 1
                    except IndexError:
                        return

        # Handle repositioning of objects
        for obj_list, new_list in [(self.doors, new_doors), (self.keys, new_keys), (self.objs, new_objs)]:
            for obj, pos in obj_list:
                new_pos = calculate_new_position(pos)
                if not is_within_bounds(new_pos):  # Find new position if out of bounds
                    new_pos = find_next_available_position(new_pos)
                    if not new_pos:
                        raise ValueError("Unable to find suitable positions for all objects when changing room size")
                new_list.append((obj, new_pos))
        new_agent_start_pos = calculate_new_position(self.agent_start_pos)
        if not is_within_bounds(new_agent_start_pos):
            new_agent_start_pos = find_next_available_position(new_agent_start_pos)
            if not new_agent_start_pos:
                raise ValueError("Unable to find suitable positions for all objects when changing room size")
        
        # Handle repositioning of walls
        if size_delta < 0:  # shrinking
            for _, (x, y) in self.walls:
                new_x, new_y = x - offset, y - offset
                if is_within_bounds((new_x, new_y)):
                    new_walls.append((Wall(), (new_x, new_y)))
        else:  # expanding
            wall_positions = [pos for _, pos in self.walls]
            # Search for and make row walls
            row_distribution = []
            new_assignments = []
            for y in range(1, self.room_size - 1):
                curr_row = [(x, y) for x in range(1, self.room_size - 1)]
                wall_detection = [cell in wall_positions for cell in curr_row]
                if any(wall_detection):
                    row_distribution.append(True)
                    new_assignments.append(calculate_new_wall_assignments(wall_detection))
                else:
                    row_distribution.append(False)
            row_sections = calculate_new_wall_positions(row_distribution)
            fill_new_walls(row_sections, new_assignments, True)
            # Search for and make  column walls
            col_distribution = []
            new_assignments = []
            for x in range(1, self.room_size - 1):
                curr_col = [(x, y) for y in range(1, self.room_size - 1)]
                wall_detection = [cell in wall_positions for cell in curr_col]
                if any(wall_detection):
                    col_distribution.append(True)
                    new_assignments.append(calculate_new_wall_assignments(wall_detection))
                else:
                    col_distribution.append(False)
            col_sections = calculate_new_wall_positions(col_distribution)
            fill_new_walls(col_sections, new_assignments, False)
            
        self.doors, self.keys, self.objs, self.walls = new_doors, new_keys, new_objs, new_walls
        self.agent_start_pos = new_agent_start_pos
        self.room_size = new_size
        self.width = new_size
        self.height = new_size
        self._gen_grid(new_size, new_size)

    
    def change_room_orientation(self, rotate_degrees: int = None):
        if not rotate_degrees:
            rotate_degrees = random.choice([90, 180, 270])
        self._gen_rotated_room(rotate_degrees)
        self._gen_grid(self.room_size, self.room_size)
    

    def change_target_color(self, new_color1 = None, new_color2 = None):
        assert self.task != Task.CLUSTER, "CLUSTER tasks do not have a singular target color to change"
        if not new_color1:
            old_color = self.target_obj.color if hasattr(self, "target_obj") else self.target_objs[0].color
            new_color1 = random.choice(list(set(OBJECT_COLOR_NAMES) - set([old_color])))
        if hasattr(self, "target_obj"):
            self.target_obj.color = new_color1
        else:
            if self.task == PUT:
                self.target_objs[0].color = new_color1
                if new_color2 is not None:
                    self.target_objs[1].color = new_color2
            else:
                for to in self.target_objs:
                    to.color = new_color1
        self._gen_grid(self.room_size, self.room_size)

    
    def hide_targets(self):
        if self.task in [Task.GOTO, Task.PICKUP]:
            if type(self.target_obj) != Box:
                box = Box(color = random.choice(OBJECT_COLOR_NAMES))
                box.contains = self.target_obj
                self.objs.remove((self.target_obj, self.target_obj_pos))
                self.objs.append((box, self.target_obj_pos))
        else:
            target_objs = flatten_list(self.target_objs)
            target_objs_pos = flatten_list(self.target_objs_pos)
            for to, top in zip(target_objs, target_objs_pos):
                if type(to) != Box:
                    box = Box(color = random.choice(OBJECT_COLOR_NAMES))
                    box.contains = to
                    self.objs.remove((to, top))
                    self.objs.append((box, top))
        self._gen_grid(self.room_size, self.room_size)
    

    def hide_keys(self):
        if len(self.keys) == 0:
            warnings.warn("Cannot hide keys in environment without keys")
            return
        for key, pos in self.keys:
            box = Box(color = random.choice(OBJECT_COLOR_NAMES))
            box.contains = key
            self.objs.remove((key, pos))
            self.objs.append((box, pos))
        self._gen_grid(self.room_size, self.room_size)
    

    def remove_keys(self):
        if len(self.keys) == 0:
            warnings.warn("Cannot remove keys in environment without keys")
            return
        self.keys = []
        for door, _ in self.doors:
            door.is_locked = False
        self._gen_grid(self.room_size, self.room_size)
    

    def change_field_of_vision(self, new_fov = None):
        if not new_fov:
            new_fov = random.choice(list(range(MIN_VIEW_SIZE, MAX_VIEW_SIZE + 1, 2)))
        self.agent_view_size = new_fov
        self._gen_grid(self.room_size, self.room_size)

    
    def toggle_doors(self):
        if len(self.doors) == 0:
            warnings.warn("Cannot toggle doors for environment without doors")
            return
        for door, _ in self.doors:
            door.is_locked = not door.is_locked
        self._gen_grid(self.room_size, self.room_size)
    
    
    """
    Private Methods
    """
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
            if isinstance(self.objs[i], list):
                for j in range(len(self.objs[i])):
                    obj, obj_pos = self.objs[i][j]
                    assert obj_pos not in wall_positions, "Object cannot be in a wall"
                    self.grid.set(obj_pos[0], obj_pos[1], obj)
                    obj.init_pos = obj_pos
            else:
                obj, obj_pos = self.objs[i]
                assert obj_pos not in wall_positions, "Object cannot be in a wall"
                self.grid.set(obj_pos[0], obj_pos[1], obj)
                obj.init_pos = obj_pos
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
    

    def _gen_rotated_room(self, degrees, ref_env = None):
        room_size = ref_env.room_size if ref_env is not None else self.room_size
        rotation_deltas = {
            90: lambda x, y: (room_size - 1 - y, x),
            180: lambda x, y: (room_size - 1 - x, room_size - 1 - y),
            270: lambda x, y: (y, room_size - 1 - x)
        }
        delta_func = rotation_deltas[degrees]

        if ref_env is None:  # editing self; no need for reference environment
            ref_objs, self.objs = self.objs, []
            ref_walls, self.walls = self.walls, []
            ref_doors, self.doors = self.doors, []
            ref_keys, self.keys = self.keys, []
            self.agent_start_pos = delta_func(*self.agent_start_pos)
            if self.task in [Task.GOTO, Task.PICKUP]:
                self.target_obj_pos = delta_func(*self.target_obj_pos)
            elif self.task in [Task.CLUSTER]:
                for i in range(len(self.target_objs_pos)):
                    for j in range(len(self.target_objs_pos[i])):
                        self.target_objs_pos[i][j] = delta_func(*self.target_objs_pos[i][j])
            else:
                self.target_objs_pos = [delta_func(*obj_pos) for obj_pos in self.target_objs_pos]
        
        else:
            ref_objs = ref_env.objs
            ref_walls = ref_env.walls
            ref_doors = ref_env.doors
            ref_keys = ref_env.keys
            self.agent_start_pos = delta_func(*ref_env.agent_start_pos)
            self.agent_start_dir = ref_env.agent_start_dir  # dir does not rotate otherwise it's just the same room
            if self.task in [Task.GOTO, Task.PICKUP]:
                self.target_obj = ref_env.target_obj
                self.target_obj_pos = delta_func(*ref_env.target_obj_pos)
            elif self.task in [Task.CLUSTER]:
                self.target_objs = ref_env.target_objs
                self.target_objs_pos = []
                for i in range(len(ref_env.target_objs_pos)):
                    objs_pos = []
                    for j in range(len(ref_env.target_objs_pos[i])):
                        objs_pos.append(delta_func(*ref_env.target_objs_pos[i][j]))
                    self.target_objs_pos.append(objs_pos)
            else:
                self.target_objs_pos = [delta_func(*obj_pos) for obj_pos in ref_env.target_objs_pos]
            if self.level == Level.MULT_ROOMS:
                self.num_rooms = ref_env.num_rooms

        for obj, pos in ref_objs:
            self.objs.append((obj, delta_func(*pos)))
        for wall, pos in ref_walls:
            self.walls.append((wall, delta_func(*pos)))
        for door, pos in ref_doors:
            self.doors.append((door, delta_func(*pos)))
        for key, pos in ref_keys:
            self.keys.append((key, delta_func(*pos)))