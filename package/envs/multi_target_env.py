from package.constants import *
from package.utils import *
from package.envs.pragmatic_env import PragmaticEnv
from package.enums import *

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box, WorldObj
from minigrid.core.mission import MissionSpace

import numpy as np
import random
from typing import List, Tuple, Dict, Any
import copy


class MultiTargetEnv(PragmaticEnv):
    def __init__(self,
                 env_type: EnvType,
                 level: Level,
                 mission_space = MissionSpace,
                 target_objs: List[WorldObj] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        super().__init__(env_type,
                         level,
                         mission_space,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
        
        if Variant.ORIENTATION in self.variants:
            self._gen_grid(self.room_size, self.room_size)
            return
        
        # Generate random environment
        self.target_objs = []
        self.target_objs_pos = []
        all_possible_pos = set([(x, y) for x in range(1, self.room_size - 1) for y in range(1, self.room_size - 1)])
        self.agent_start_dir = np.random.randint(0, 4)

        if level in [Level.EMPTY, Level.DIST]:
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
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            if level == Level.DIST:
                if Variant.NUM_OBJECTS in self.variants:
                    num_distractors = random.choice(list(set(range(1, self.room_size - 3)) - set([self.disallowed[Variant.NUM_OBJECTS]])))
                else:
                    num_distractors = np.random.choice(range(1, self.room_size - 3))
                disallowed_obj_config = set([(type(obj), obj.color) for obj in flatten_list(self.target_objs)])
                if Variant.OBJECTS in self.variants:
                    disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
                    required_obj_positions = self.disallowed[Variant.OBJECTS][1]
                    num_distractors = len(required_obj_positions)
                for i in range(num_distractors):
                    temp = flatten_list(self.target_objs)
                    dist_obj = temp[0]
                    while (type(dist_obj), dist_obj.color) in disallowed_obj_config:
                        dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                    if Variant.OBJECTS in self.variants and required_obj_positions[i] in all_possible_pos:
                        dist_obj_pos = required_obj_positions[i]
                    else:
                        dist_obj_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
        
        elif level in [Level.DEATH]:
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
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            path_to_target = self._gen_path_to_target()
            all_possible_pos -= path_to_target
            num_lavas = min(len(all_possible_pos) - 1, random.choice(range(int(0.25 * (self.room_size - 2)**2), int(0.4 * (self.room_size - 2)**2))))
            all_possible_pos = list(all_possible_pos)
            lava_positions = np.random.choice(len(all_possible_pos), num_lavas, replace = False)
            for p in lava_positions:
                self.objs.append((Lava(), all_possible_pos[p]))
        
        elif level in [Level.OPEN_DOOR, Level.BLOCKED_DOOR, Level.UNLOCK_DOOR, Level.HIDDEN_KEY]:
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
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            temp = flatten_list(self.target_objs_pos)
            if wall_orientation == "vertical":
                if temp[0][0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([temp[0][0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([temp[0][0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
                self.walls = [(Wall(), (wall_col, y)) for y in range(1, self.room_size - 1)]
            elif wall_orientation == "horizontal":
                if temp[0][1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([temp[0][1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([temp[0][1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
                self.walls = [(Wall(), (x, wall_row)) for x in range(1, self.room_size - 1)]
            wall_positions = [wall[1] for wall in self.walls]
            all_possible_pos -= set(wall_positions)
            if self.env_type == EnvType.CLUSTER:
                target_objs_pos_to_remove = []
                for i in range(len(self.target_objs_pos)):
                    for j in range(len(self.target_objs_pos[i])):
                        if self.target_objs_pos[i][j] in wall_positions:
                            target_objs_pos_to_remove.append((i, j))
                all_possible_pos = list(all_possible_pos)
                replacement_position_idx = np.random.choice(len(all_possible_pos), len(target_objs_pos_to_remove), replace = False)
                replacement_positions = []
                for rp in replacement_position_idx:
                    replacement_positions.append(all_possible_pos[rp])
                rp_idx = 0
                for i, j in target_objs_pos_to_remove:
                    self.target_objs_pos[i][j] = replacement_positions[rp_idx]
                    rp_idx += 1
                all_possible_pos = set(all_possible_pos)
                all_possible_pos -= set(replacement_positions)
            else:
                target_objs_pos_to_remove = []
                for i in range(len(self.target_objs_pos)):
                    if self.target_objs_pos[i] in wall_positions:
                        target_objs_pos_to_remove.append(i)
                all_possible_pos = list(all_possible_pos)
                replacement_position_idx = np.random.choice(len(all_possible_pos), len(target_objs_pos_to_remove), replace = False)
                replacement_positions = []
                for rp in replacement_position_idx:
                    replacement_positions.append(all_possible_pos[rp])
                rp_idx = 0
                for idx in target_objs_pos_to_remove:
                    self.target_objs_pos[idx] = replacement_positions[rp_idx]
                    rp_idx += 1
                all_possible_pos = set(all_possible_pos)
                all_possible_pos -= set(replacement_positions)
            if len(target_objs_pos_to_remove) > 0:
                self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            self.doors = [(Door(is_locked = level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY], color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
            if level == Level.BLOCKED_DOOR:
                if wall_orientation == "vertical":
                    if temp[0][0] > self.room_size // 2:
                        blocker_obj_pos = (self.doors[0][1][0] - 1, self.doors[0][1][1])
                    else:
                        blocker_obj_pos = (self.doors[0][1][0] + 1, self.doors[0][1][1])
                else:
                    if temp[0][1] > self.room_size // 2:
                        blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] - 1)
                    else:
                        blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] + 1)
                all_possible_pos -= set([blocker_obj_pos])
                temp = flatten_list(self.target_objs)
                blocker_obj = temp[0]
                disallowed_blocker_obj_config = set([(type(temp[0]), temp[0].color)])
                if Variant.OBJECTS in self.variants:
                    disallowed_blocker_obj, disallowed_blocker_color = self.disallowed[Variant.OBJECTS][0][-1]
                    disallowed_blocker_obj_config.add((disallowed_blocker_obj, disallowed_blocker_color))
                while (type(blocker_obj), blocker_obj.color) in disallowed_blocker_obj_config:
                    blocker_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                self.objs.append((blocker_obj, blocker_obj_pos))
            elif level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY]:
                key = Key(color = self.doors[0][0].color)
                key_pos = (np.random.randint(other_side_x_lb, other_side_x_ub), np.random.randint(other_side_y_lb, other_side_y_ub))
                all_possible_pos -= set([key_pos])
                if level == Level.HIDDEN_KEY:
                    if type(self.target_objs[0]) == Box or type(self.target_objs[1]) == Box:
                        box = Box(color = random.choice(list(set(OBJECT_COLOR_NAMES) - set([obj.color for obj in self.target_objs]))))
                    else:
                        box = Box(color = random.choice(OBJECT_COLOR_NAMES))
                    box.contains = key
                    self.objs.append((box, key_pos))
                else:
                    self.keys.append((key, key_pos))
            self.agent_start_pos = (np.random.randint(other_side_x_lb, other_side_x_ub), np.random.randint(other_side_y_lb, other_side_y_ub))
            while self.agent_start_pos not in all_possible_pos:
                self.agent_start_pos = (np.random.randint(other_side_x_lb, other_side_x_ub), np.random.randint(other_side_y_lb, other_side_y_ub))
        
        elif level in [Level.GO_AROUND]:
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
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            temp = flatten_list(self.target_objs_pos)
            if wall_orientation == "vertical":
                if temp[0][0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([temp[0][0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([temp[0][0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
                if temp[0][1] > self.room_size // 2:
                    wall_head = random.choice(range(2, temp[0][1]))
                    wall_tail = self.room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(temp[0][1] + 1, self.room_size - 1))
                self.walls = [(Wall(), (wall_col, y)) for y in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(other_side_x_lb, other_side_x_ub)), random.choice(range(wall_head, wall_tail)))
            elif wall_orientation == "horizontal":
                if temp[0][1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([temp[0][1]])))
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([temp[0][1]])))
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
                if temp[0][0] > self.room_size // 2:
                    wall_head = random.choice(range(2, temp[0][0]))
                    wall_tail = self.room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(temp[0][0] + 1, self.room_size - 1))
                self.walls = [(Wall(), (x, wall_row)) for x in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(wall_head, wall_tail)), random.choice(range(other_side_y_lb, other_side_y_ub)))
            self.walls = [(Wall(), wall_pos) for wall, wall_pos in self.walls if wall_pos not in temp]
            wall_positions = [wall[1] for wall in self.walls]
            all_possible_pos -= set(wall_positions)
            all_possible_pos -= set([self.agent_start_pos])
            self.doors = [(Door(color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
        
        elif level in [Level.MULT_ROOMS]:
            rooms_range = list(range(2, 5 if self.room_size <= 9 else 7))
            if Variant.NUM_ROOMS in self.variants:
                self.num_rooms = random.choice(list(set(rooms_range) - set([self.disallowed[Variant.NUM_ROOMS]])))
            else:
                self.num_rooms = random.choice(rooms_range)
            room_walls, room_doors, room_cells = self._gen_multiple_rooms()
            self.walls.extend([(Wall(), pos) for pos in room_walls + room_doors])
            self.doors.extend([(Door(is_locked = False, color = random.choice(OBJECT_COLOR_NAMES)), pos) for pos in room_doors])
            all_possible_pos -= set(room_walls)
            all_possible_pos -= set(room_doors)
            self.agent_start_pos = random.choice(list(all_possible_pos))
            for i in range(len(room_cells)):
                if self.agent_start_pos in room_cells[i]:
                    all_possible_pos -= room_cells[i]
                    break
            all_possible_pos -= set([self.agent_start_pos])
            if self.env_type == EnvType.PUT:
                positions = np.random.choice(len(all_possible_pos), 2, replace = False)
                all_possible_pos = list(all_possible_pos)
                self.target_objs_pos = [all_possible_pos[positions[0]], all_possible_pos[positions[1]]]
                all_possible_pos = set(all_possible_pos)
                all_possible_pos -= set(self.target_objs_pos)
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
        
        elif level in [Level.BOSS]:
            # Handle the MULT_ROOMS characteristics
            rooms_range = list(range(2, 5 if self.room_size <= 9 else 7))
            if Variant.NUM_ROOMS in self.variants:
                self.num_rooms = random.choice(list(set(rooms_range) - set([self.disallowed[Variant.NUM_ROOMS]])))
            else:
                self.num_rooms = random.choice(rooms_range)
            room_walls, room_doors, room_cells = self._gen_multiple_rooms()
            self.walls.extend([(Wall(), pos) for pos in room_walls + room_doors])
            # Handle the UNLOCK_DOOR characteristics
            necessary_key_colors = []
            locked_doors = 0
            for room_door_pos in room_doors:
                is_locked = random.choice([True, False])
                door = Door(is_locked = is_locked and locked_doors < MAX_NUM_LOCKED_DOORS, color = random.choice(OBJECT_COLOR_NAMES))
                if is_locked:
                    necessary_key_colors.append(door.color)
                    locked_doors += 1
                self.doors.append((door, room_door_pos))
            all_possible_pos -= set(room_walls)
            all_possible_pos -= set(room_doors)
            self.agent_start_pos = random.choice(list(all_possible_pos))
            all_possible_pos -= set([self.agent_start_pos])
            for i in range(len(room_cells)):
                if self.agent_start_pos in room_cells[i]:
                    for key_color in necessary_key_colors:
                        key_pos = random.choice(list(room_cells[i]))
                        room_cells[i].remove(key_pos)
                        self.keys.append((Key(color = key_color), key_pos))
                    all_possible_pos -= room_cells[i]
                    agent_cell_idx = i
                    break
            del room_cells[agent_cell_idx]
            # Set the target objects
            if self.env_type == EnvType.PUT:
                positions = np.random.choice(len(all_possible_pos), 2, replace = False)
                all_possible_pos = list(all_possible_pos)
                self.target_objs_pos = [all_possible_pos[positions[0]], all_possible_pos[positions[1]]]
                all_possible_pos = set(all_possible_pos)
                all_possible_pos -= set(self.target_objs_pos)
            elif self.env_type in [EnvType.COLLECT, EnvType.CLUSTER]:
                if self.env_type == EnvType.COLLECT:
                    num_collectibles = random.choice(range(2, 5))
                else:
                    num_collectibles = random.choice(range(3, 8))
                for _ in range(num_collectibles):
                    collectible_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([collectible_pos])
                    all_possible_pos -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
            self._set_target_objs(target_objs)
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
            # Handle the DIST and DEATH characteristics
            smallest_room = len(min(room_cells, key = len))
            if Variant.NUM_OBJECTS in self.variants:
                num_distractors = random.choice(list(set(range(1, smallest_room)) - set([self.disallowed[Variant.NUM_OBJECTS]])))
            else:
                num_distractors = np.random.choice(range(1, smallest_room))
            disallowed_obj_config = set([(type(obj), obj.color) for obj in flatten_list(self.target_objs)])
            if Variant.OBJECTS in self.variants:
                disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
                required_obj_positions = self.disallowed[Variant.OBJECTS][1]
                num_distractors = min(smallest_room, len(required_obj_positions))
            for i in range(num_distractors):
                actually_lava = random.choice([True, False])
                if actually_lava:
                    if Variant.OBJECTS in self.variants and required_obj_positions[i] in all_possible_pos:
                        lava_pos = required_obj_positions[i]
                    else:
                        lava_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([lava_pos])
                    self.objs.append((Lava(), lava_pos))
                else:
                    temp = flatten_list(self.target_objs)
                    dist_obj = temp[0]
                    while (type(dist_obj), dist_obj.color) in disallowed_obj_config:
                        dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                    if Variant.OBJECTS in self.variants and required_obj_positions[i] in all_possible_pos:
                        dist_obj_pos = required_obj_positions[i]
                    else:
                        dist_obj_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
        
        self.agent_start_dir = np.random.randint(0, 4)
        self._gen_grid(self.room_size, self.room_size)
            
        # Set mission
        if self.env_type == EnvType.PUT:
            self.mission = f"put the {self.target_objs[0].color} {OBJ_NAME_MAPPING[type(self.target_objs[0])]} next to the {self.target_objs[1].color} {OBJ_NAME_MAPPING[type(self.target_objs[1])]}"
        elif self.env_type == EnvType.COLLECT:
            self.mission = f"put all {self.target_objs[0].color} {OBJ_PLURAL_MAPPING[type(self.target_objs[0])]} next to each other"
        elif self.env_type == EnvType.CLUSTER:
            if self.env_seed % 2 == 1:
                self.mission = f"group all {self.target_objs[0].color} objects by type"
            else:
                self.mission = f"group all {OBJ_PLURAL_MAPPING[type(self.target_objs[0][0])]} by color"
        
        # Final asserts
        assert self.agent_start_pos is not None, "self.agent_start_pos is None"
        assert self.agent_start_dir is not None, "self.agent_start_dir is None"
        assert len(self.target_objs) != 0, "self.target_objs is empty"
        assert len(self.target_objs_pos) != 0, "self.target_objs_pos is empty"
    
    
    def _set_target_objs(self, target_objs):
        if target_objs is None:
            if Variant.COLOR in self.variants:
                disallowed_color = self.disallowed[Variant.COLOR]
            else:
                disallowed_color = ""
            allowed_colors = [color for color in OBJECT_COLOR_NAMES if color != disallowed_color]
            if self.env_type == EnvType.PUT:
                idx1 = self.env_seed % len(TANGIBLE_OBJS)
                idx2 = (self.env_seed + 1) % len(PLAYABLE_OBJS)
                color1 = random.choice(allowed_colors)
                allowed_colors.remove(color1)
                color2 = random.choice(allowed_colors)
                pickup_obj = TANGIBLE_OBJS[idx1](color = color1)
                target_obj = PLAYABLE_OBJS[idx2]
                if target_obj == Goal:
                    target_obj = Goal()
                    target_obj.color = color2
                else:
                    target_obj = target_obj(color = color2)
                self.target_objs = [pickup_obj, target_obj]
            elif self.env_type == EnvType.COLLECT:
                collectible_obj = TANGIBLE_OBJS[self.env_seed % len(TANGIBLE_OBJS)]
                color = random.choice(allowed_colors)
                for _ in range(len(self.target_objs_pos)):
                    self.target_objs.append(collectible_obj(color = color))
            elif self.env_type == EnvType.CLUSTER:
                if len(self.target_objs) <= 5:
                    num_clusters = 2
                else:
                    num_clusters = random.choice([2, 3])
                clustered_pos = make_clusters(self.target_objs_pos, num_clusters)
                if self.env_seed % 2 == 1:
                    target_color = random.choice(allowed_colors)
                    for i in range(num_clusters):
                        this_cluster = []
                        for _ in range(len(clustered_pos[i])):
                            random_obj = random.choice(TANGIBLE_OBJS)
                            this_cluster.append(random_obj(color = target_color))
                        self.target_objs.append(this_cluster)
                else:
                    target_obj = TANGIBLE_OBJS[self.env_seed % len(TANGIBLE_OBJS)]
                    for i in range(num_clusters):
                        this_cluster = []
                        for _ in range(len(clustered_pos[i])):
                            random_obj = target_obj(color = random.choice(allowed_colors))
                            this_cluster.append(random_obj)
                        self.target_objs.append(this_cluster)
                self.target_objs_pos = clustered_pos
        else:
            if self.env_type == EnvType.CLUSTER:
                reference = copy.deepcopy(self.target_objs_pos)
                self.target_objs_pos = []
                pos_idx = 0
                for obj_cluster in target_objs:
                    this_cluster = []
                    this_cluster.extend([obj(color = random.choice(OBJECT_COLOR_NAMES)) for obj in obj_cluster])
                    self.target_objs.append(this_cluster)
                    self.target_objs_pos.append(reference[pos_idx : pos_idx + len(this_cluster)])
                    pos_idx += len(this_cluster)
            else:
                self.target_objs.extend([obj(color = random.choice(OBJECT_COLOR_NAMES)) for obj in target_objs])
