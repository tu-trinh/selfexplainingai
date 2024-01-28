from package.constants import *
from package.envs.pragmatic_env import PragmaticEnv
from package.enums import *
from package.utils import *

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
from minigrid.core.mission import MissionSpace

import numpy as np
import random
from typing import List, Tuple, Dict, Any


class SingleTargetEnv(PragmaticEnv):
    def __init__(self,
                 env_type: EnvType,
                 level: Level,
                 mission_space: MissionSpace,
                 target_obj: Tuple[PLAYABLE_OBJS] = None,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        super().__init__(env_type,
                         level,
                         mission_space,
                         max_steps = max_steps,
                         see_through_walls = see_through_walls,
                         **kwargs)
        
        self.target_obj = None
        self.target_obj_pos = None
        all_possible_pos = set([(x, y) for x in range(1, self.room_size - 1) for y in range(1, self.room_size - 1)])
        self.agent_start_dir = np.random.randint(0, 4)

        if level in [Level.EMPTY, Level.DIST]:
            all_possible_pos = list(all_possible_pos)
            positions = np.random.choice(len(all_possible_pos), 2, replace = False)
            self.agent_start_pos, self.target_obj_pos = all_possible_pos[positions[0]], all_possible_pos[positions[1]]
            all_possible_pos = set(all_possible_pos)
            all_possible_pos -= set([self.agent_start_pos, self.target_obj_pos])
            self._set_target_obj(target_obj)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            if level == Level.DIST:
                if Variant.NUM_OBJECTS in self.variants:
                    num_distractors = random.choice(list(set(range(1, self.room_size - 3)) - set([self.disallowed[Variant.NUM_OBJECTS]])))
                else:
                    num_distractors = np.random.choice(range(1, self.room_size - 3))
                disallowed_obj_config = set([(type(self.target_obj), self.target_obj.color)])
                if Variant.OBJECTS in self.variants:
                    disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
                    required_obj_positions = self.disallowed[Variant.OBJECTS][1]
                    num_distractors = len(required_obj_positions)
                for i in range(num_distractors):
                    dist_obj = self.target_obj
                    while (type(dist_obj), dist_obj.color) in disallowed_obj_config:
                        dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                    if Variant.OBJECTS in self.variants and required_obj_positions[i] in all_possible_pos:
                        dist_obj_pos = required_obj_positions[i]
                    else:
                        dist_obj_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
        
        elif level in [Level.DEATH]:
            all_possible_pos = list(all_possible_pos)
            positions = np.random.choice(len(all_possible_pos), 2, replace = False)
            self.agent_start_pos, self.target_obj_pos = all_possible_pos[positions[0]], all_possible_pos[positions[1]]
            all_possible_pos = set(all_possible_pos)
            all_possible_pos -= set([self.agent_start_pos, self.target_obj_pos])
            self._set_target_obj(target_obj)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            path_to_target = self._gen_path_to_target()
            all_possible_pos -= set(path_to_target)
            num_lavas = random.choice(range(int(0.25 * (self.room_size - 2)**2), int(0.4 * (self.room_size - 2)**2)))
            all_possible_pos = list(all_possible_pos)
            lava_positions = np.random.choice(len(all_possible_pos), num_lavas, replace = False)
            for p in lava_positions:
                self.objs.append((Lava(), all_possible_pos[p]))
        
        elif level in [Level.OPEN_DOOR, Level.BLOCKED_DOOR, Level.UNLOCK_DOOR, Level.HIDDEN_KEY]:
            self.target_obj_pos = random.choice(list(all_possible_pos))
            all_possible_pos -= set([self.target_obj_pos])
            self._set_target_obj(target_obj)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            if wall_orientation == "vertical":
                if self.target_obj_pos[0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
                self.walls = [(Wall(), (wall_col, y)) for y in range(1, self.room_size - 1)]
            elif wall_orientation == "horizontal":
                if self.target_obj_pos[1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([self.target_obj_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([self.target_obj_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
                self.walls = [(Wall(), (x, wall_row)) for x in range(1, self.room_size - 1)]
            wall_positions = [wall[1] for wall in self.walls]
            self.doors = [(Door(is_locked = level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY], color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
            all_possible_pos -= set(wall_positions)
            if level == Level.BLOCKED_DOOR:
                if wall_orientation == "vertical":
                    if self.target_obj_pos[0] > self.room_size // 2:
                        blocker_obj_pos = (self.doors[0][1][0] - 1, self.doors[0][1][1])
                    else:
                        blocker_obj_pos = (self.doors[0][1][0] + 1, self.doors[0][1][1])
                else:
                    if self.target_obj_pos[1] > self.room_size // 2:
                        blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] - 1)
                    else:
                        blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] + 1)
                all_possible_pos -= set([blocker_obj_pos])
                blocker_obj = self.target_obj
                while type(blocker_obj) == type(self.target_obj) and blocker_obj.color == self.target_obj.color:
                    blocker_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                self.objs.append((blocker_obj, blocker_obj_pos))
            elif level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY]:
                key = Key(color = self.doors[0][0].color)
                key_pos = (np.random.randint(other_side_x_lb, other_side_x_ub), np.random.randint(other_side_y_lb, other_side_y_ub))
                all_possible_pos -= set([key_pos])
                if level == Level.HIDDEN_KEY:
                    if type(self.target_obj) == Box:
                        box = Box(color = random.choice(list(set(OBJECT_COLOR_NAMES) - set([self.target_obj.color]))))
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
            self.target_obj_pos = random.choice(list(all_possible_pos))
            all_possible_pos -= set([self.target_obj_pos])
            self._set_target_obj(target_obj)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            if wall_orientation == "vertical":
                if self.target_obj_pos[0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
                if self.target_obj_pos[1] > self.room_size // 2:
                    wall_head = random.choice(range(2, self.target_obj_pos[1]))
                    wall_tail = self.room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(self.target_obj_pos[1] + 1, self.room_size - 1))
                self.walls = [(Wall(), (wall_col, y)) for y in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(other_side_x_lb, other_side_x_ub)), random.choice(range(wall_head, wall_tail)))
            elif wall_orientation == "horizontal":
                if self.target_obj_pos[1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([self.target_obj_pos[1]])))
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([self.target_obj_pos[1]])))
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
                if self.target_obj_pos[0] > self.room_size // 2:
                    wall_head = random.choice(range(2, self.target_obj_pos[0]))
                    wall_tail = self.room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(self.target_obj_pos[0] + 1, self.room_size - 1))
                self.walls = [(Wall(), (x, wall_row)) for x in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(wall_head, wall_tail)), random.choice(range(other_side_y_lb, other_side_y_ub)))
            wall_positions = [wall[1] for wall in self.walls]
            all_possible_pos -= set(wall_positions)
            all_possible_pos -= set([self.agent_start_pos])
            self.doors = [(Door(color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
        
        elif level in [Level.MULT_ROOMS]:
            self.num_rooms = random.randint(2, self.room_size - 3 if self.room_size % 2 == 1 else self.room_size - 4)
            self._gen_multiple_rooms()
        
        self._gen_grid(self.room_size, self.room_size)
            
        # Set mission
        if self.env_type == EnvType.GOTO:
            self.mission = f"get to the {OBJ_NAME_MAPPING[type(self.target_obj)]}"
        elif self.env_type == EnvType.PICKUP:
            self.mission = f"pick up the {OBJ_NAME_MAPPING[type(self.target_obj)]}"
        
        # Final asserts
        assert self.agent_start_pos is not None, "self.agent_start_pos is None"
        assert self.agent_start_dir is not None, "self.agent_start_dir is None"
        assert self.target_obj is not None, "self.target_obj is None"
        assert self.target_obj_pos is not None, "self.target_obj_pos is None"
    
    
    def _set_target_obj(self, target_obj):
        if target_obj is None:
            if self.env_type == EnvType.GOTO:
                index = self.env_seed % len(PLAYABLE_OBJS)
                target_obj = PLAYABLE_OBJS[index]
            elif self.env_type == EnvType.PICKUP:
                index = self.env_seed % len(TANGIBLE_OBJS)
                target_obj = TANGIBLE_OBJS[index]
        if Variant.COLOR in self.variants:
            color = random.choice(list(set(OBJECT_COLOR_NAMES) - set([self.disallowed[Variant.COLOR]])))
        else:
            color = random.choice(OBJECT_COLOR_NAMES)
        if target_obj == Goal:
            self.target_obj = Goal()
            self.target_obj.color = color
        else:
            self.target_obj = target_obj(color = color)
    
    
    def _gen_path_to_target(self):
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
                if pos == self.target_obj_pos:
                    reached_object = True
                    break
            direction = min(3, direction + 1) if np.random.random() > 0.5 else max(0, direction - 1)
            num_turns += 1
        if pos != self.target_obj_pos:
            if pos[0] < self.target_obj_pos[0]:
                horizontal_step = 1
            elif pos[0] > self.target_obj_pos[0]:
                horizontal_step = -1
            else:
                horizontal_step = None
            if pos[1] < self.target_obj_pos[1]:
                vertical_step = 1
            elif pos[1] > self.target_obj_pos[1]:
                vertical_step = -1
            else:
                vertical_step = None
            if horizontal_step:
                for x in range(pos[0] + horizontal_step, self.target_obj_pos[0] + horizontal_step, horizontal_step):
                    new_pos = (x, pos[1])
                    path.append(new_pos)
                    pos = new_pos
            if vertical_step:
                for y in range(pos[1] + vertical_step, self.target_obj_pos[1] + vertical_step, vertical_step):
                    new_pos = (pos[0], y)
                    path.append(new_pos)
                    pos = new_pos
        return path
