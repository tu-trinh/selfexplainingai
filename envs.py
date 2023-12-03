from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from typing import Tuple, List, Union
from enum import Enum
from constants import *
from minigrid.manual_control import ManualControl
import numpy as np
import random
import gymnasium
from utils import *
import copy


class Level(Enum):
    """
    EMPTY: Completely empty room except for the necessary object(s) to complete mission
    DEATH: Presence of lava that will kill the agent if touched
    DIST_SAME: Presence of distractors of the same type as the mission object(s) but different colors
    DIST_DIFF: Presence of random distractors of all types and colors
    OPEN_DOOR: Must open an unlocked door at some point to complete mission
    UNLOCK_DOOR: Must find a key to unlock and open a door at some point to complete mission
    GO_AROUND: Must go around a line of walls or some other blocking object at some point
    MULT_ROOMS: Multiple rooms with doors of various locked/unlocked states
    BOSS: Combine MULT_ROOMS, DIST_SAME, DIST_DIFF, and BLOCKED_DOOR
    """
    EMPTY = "Empty"
    DEATH = "Death"
    DIST_SAME = "DistSame"
    DIST_DIFF = "DistDiff"
    OPEN_DOOR = "OpenDoor"
    BLOCKED_DOOR = "BlockedDoor"
    UNLOCK_DOOR = "UnlockDoor"
    HIDDEN_KEY = "HiddenKey"
    GO_AROUND = "GoAround"
    MULT_ROOMS = "MultRooms"
    BOSS = "Boss"


class EnvType(Enum):
    GOTO = "Goto"
    PICKUP = "Pickup"
    PUT = "Put"
    ARRANGE = "Arrange"
    COLLECT = "Collect"


class EnvMaker:
    def generate_env(self,
                     env_type: EnvType,
                     env_seed: int,
                     level: Level,
                     max_steps: int = None,
                     see_through_walls = False,
                     **kwargs):
        if env_type == EnvType.GOTO:
            if level == Level.MULT_ROOMS:
                choices = ["MiniGrid-LockedRoom-v0"]
                env = gymnasium.make(random.choice(choices), render_mode = "human")
            elif level == Level.BOSS:
                choices = ["MiniGrid-Playground-v0"]
                env = gymnasium.make(random.choice(choices), render_mode = "human")
            else:
                env = GotoEnv(env_seed, level, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
        elif env_type == EnvType.PICKUP:
            if level == Level.MULT_ROOMS:
                choices = ["BabyAI-UnlockToUnlock-v0"]
                env = gymnasium.make(random.choice(choices), render_mode = "human")
            elif level == Level.BOSS:
                choices = ["MiniGrid-Playground-v0"]
                env = gymnasium.make(random.choice(choices), render_mode = "human")
            else:
                env = PickupEnv(env_seed, level, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
        return env


class SingleTargetEnv(MiniGridEnv):
    def __init__(self,
                 env_type: EnvType,
                 env_seed: int,
                 level: Level,
                 mission_space: MissionSpace,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        assert env_type in [EnvType.GOTO, EnvType.PICKUP], "Env type must either be Goto or Pickup"
        assert level in [Level.EMPTY, Level.DIST_DIFF, Level.OPEN_DOOR, Level.BLOCKED_DOOR, Level.UNLOCK_DOOR,
                         Level.HIDDEN_KEY, Level.GO_AROUND, Level.DEATH]
        np.random.seed(env_seed)
        random.seed(env_seed)
        self.env_type = env_type
        
        # Super init
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        if max_steps is None:
            self.max_steps = 4 * room_size ** 2
        else:
            self.max_steps = max_steps
        super().__init__(mission_space = mission_space, grid_size = room_size, max_steps = self.max_steps,
                         see_through_walls = see_through_walls, render_mode = "human", agent_view_size = AGENT_VIEW_SIZE, **kwargs)
        
        # Generate random environment
        self.env_id = f"{self.env_type}-{level}-{env_seed}"
        self.objs = []  # List of (object, (x, y))
        self.walls = []  # List of (wall, x, y)
        self.doors = []  # List of (door, (x, y))
        self.keys = []  # List of (key, x, y)
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.target_obj = None
        self.target_obj_pos = None
        all_possible_pos = set([(x, y) for x in range(1, room_size - 1) for y in range(1, room_size - 1)])

        if level in [Level.EMPTY, Level.DIST_DIFF]:
            all_possible_pos = list(all_possible_pos)
            positions = np.random.choice(len(all_possible_pos), 2, replace = False)
            self.agent_start_pos, self.target_obj_pos = all_possible_pos[positions[0]], all_possible_pos[positions[1]]
            all_possible_pos = set(all_possible_pos)
            all_possible_pos -= set([self.agent_start_pos, self.target_obj_pos])
            self._set_target_obj(env_seed)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            if level == Level.DIST_DIFF:
                num_distractors = np.random.choice(range(1, room_size - 3))
                for _ in range(num_distractors):
                    dist_obj = self.target_obj
                    while type(dist_obj) == type(self.target_obj) and dist_obj.color == self.target_obj.color:
                        dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                    dist_obj_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
            self._gen_grid(room_size, room_size)
        
        elif level in [Level.DEATH]:
            all_possible_pos = list(all_possible_pos)
            positions = np.random.choice(len(all_possible_pos), 2, replace = False)
            self.agent_start_pos, self.target_obj_pos = all_possible_pos[positions[0]], all_possible_pos[positions[1]]
            all_possible_pos = set(all_possible_pos)
            all_possible_pos -= set([self.agent_start_pos, self.target_obj_pos])
            self._set_target_obj(env_seed)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            path_to_target = self._gen_path_to_target(room_size)
            all_possible_pos -= set(path_to_target)
            num_lavas = random.choice(range(int(0.25 * (room_size - 2)**2), int(0.4 * (room_size - 2)**2)))
            all_possible_pos = list(all_possible_pos)
            lava_positions = np.random.choice(len(all_possible_pos), num_lavas, replace = False)
            for p in lava_positions:
                self.objs.append((Lava(), all_possible_pos[p]))
        
        elif level in [Level.OPEN_DOOR, Level.BLOCKED_DOOR, Level.UNLOCK_DOOR, Level.HIDDEN_KEY]:
            self.target_obj_pos = random.choice(list(all_possible_pos))
            all_possible_pos -= set([self.target_obj_pos])
            self._set_target_obj(env_seed)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            if wall_orientation == "vertical":
                if self.target_obj_pos[0] > room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                    other_side_y_lb, other_side_y_ub = 1, room_size - 1
                else:
                    wall_col = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, room_size - 1
                self.walls = [(Wall(), (wall_col, y)) for y in range(1, room_size - 1)]
            elif wall_orientation == "horizontal":
                if self.target_obj_pos[1] > room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([self.target_obj_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([self.target_obj_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, room_size - 1
                    other_side_y_lb, other_side_y_ub = wall_row + 1, room_size - 1
                self.walls = [(Wall(), (x, wall_row)) for x in range(1, room_size - 1)]
            wall_positions = [wall[1] for wall in self.walls]
            self.doors = [(Door(is_locked = level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY], color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
            all_possible_pos -= set(wall_positions)
            if level == Level.BLOCKED_DOOR:
                if wall_orientation == "vertical":
                    if self.target_obj_pos[0] > room_size // 2:
                        blocker_obj_pos = (self.doors[0][1][0] - 1, self.doors[0][1][1])
                    else:
                        blocker_obj_pos = (self.doors[0][1][0] + 1, self.doors[0][1][1])
                else:
                    if self.target_obj_pos[1] > room_size // 2:
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
            self._set_target_obj(env_seed)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
            if wall_orientation == "vertical":
                if self.target_obj_pos[0] > room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                else:
                    wall_col = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([self.target_obj_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, room_size - 1
                if self.target_obj_pos[1] > room_size // 2:
                    wall_head = random.choice(range(2, self.target_obj_pos[1]))
                    wall_tail = room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(self.target_obj_pos[1] + 1, room_size - 1))
                self.walls = [(Wall(), (wall_col, y)) for y in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(other_side_x_lb, other_side_x_ub)), random.choice(range(wall_head, wall_tail)))
            elif wall_orientation == "horizontal":
                if self.target_obj_pos[1] > room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, room_size // 2 + 1)) - set([self.target_obj_pos[1]])))
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(room_size // 2 + 1, room_size - 2)) - set([self.target_obj_pos[1]])))
                    other_side_y_lb, other_side_y_ub = wall_row + 1, room_size - 1
                if self.target_obj_pos[0] > room_size // 2:
                    wall_head = random.choice(range(2, self.target_obj_pos[0]))
                    wall_tail = room_size - 1
                else:
                    wall_head = 1
                    wall_tail = random.choice(range(self.target_obj_pos[0] + 1, room_size - 1))
                self.walls = [(Wall(), (x, wall_row)) for x in range(wall_head, wall_tail)]
                self.agent_start_pos = (random.choice(range(wall_head, wall_tail)), random.choice(range(other_side_y_lb, other_side_y_ub)))
            wall_positions = [wall[1] for wall in self.walls]
            all_possible_pos -= set(wall_positions)
            all_possible_pos -= set([self.agent_start_pos])
            self.doors = [(Door(color = random.choice(OBJECT_COLOR_NAMES)), random.choice(wall_positions))]
        
        self.agent_start_dir = np.random.randint(0, 4)
            
        # Set mission
        if self.env_type == EnvType.GOTO:
            self.mission = f"get to the {self.target_obj.color} {OBJ_NAME_MAPPING[type(self.target_obj)]}"
        elif self.env_type == EnvType.PICKUP:
            self.mission = f"pick up the {self.target_obj.color} {OBJ_NAME_MAPPING[type(self.target_obj)]}"
        
        # Final asserts
        assert self.agent_start_pos is not None, "self.agent_start_pos is None"
        assert self.agent_start_dir is not None, "self.agent_start_dir is None"
        assert self.target_obj is not None, "self.target_obj is None"
        assert self.target_obj_pos is not None, "self.target_obj_pos is None"
    
    def _set_target_obj(self, env_seed):
        if self.env_type == EnvType.GOTO:
            index = env_seed % len(GOTO_TARGET_OBJS)
            target_obj = GOTO_TARGET_OBJS[index]
        elif self.env_type == EnvType.PICKUP:
            index = env_seed % len(PICKUP_TARGET_OBJS)
            target_obj = PICKUP_TARGET_OBJS[index]
        color = OBJECT_COLOR_NAMES[env_seed % len(OBJECT_COLOR_NAMES)]
        if target_obj == Goal:
            self.target_obj = Goal()
            self.target_obj.color = color
        else:
            self.target_obj = target_obj(color = color)
    
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
    
    def _gen_path_to_target(self, room_size):
        path = [tuple(self.agent_start_pos)]
        current_pos = self.agent_start_pos
        num_turns = random.randint(1, room_size - 4)
        for _ in range(num_turns):
            if current_pos == self.target_obj_pos:
                break
            directions = []
            if current_pos[0] < room_size - 2:
                directions.append((1, 0))
            if current_pos[0] > 1:
                directions.append((-1, 0))
            if current_pos[1] < room_size - 2:
                directions.append((0, 1))
            if current_pos[1] > 1:
                directions.append((0, -1))
            if not directions:
                continue
            direction = random.choice(directions)
            if direction == (1, 0):
                max_steps = room_size - 2 - current_pos[0]
            elif direction == (-1, 0):
                max_steps = current_pos[0] - 1
            elif direction == (0, 1):
                max_steps = room_size - 2 - current_pos[1]
            elif direction == (0, -1):
                max_steps = current_pos[1] - 1
            steps = random.randint(1, max_steps)
            for __ in range(steps):
                new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                path.append(new_pos)
                current_pos = new_pos
                if current_pos == self.target_obj_pos:
                    break
        while current_pos != self.target_obj_pos:
            if current_pos[0] < min(self.target_obj_pos[0], room_size - 2):
                new_pos = (current_pos[0] + 1, current_pos[1])
            elif current_pos[0] > max(self.target_obj_pos[0], 1):
                new_pos = (current_pos[0] - 1, current_pos[1])
            elif current_pos[1] < min(self.target_obj_pos[1], room_size - 2):
                new_pos = (current_pos[0], current_pos[1] + 1)
            elif current_pos[1] > max(self.target_obj_pos[1], 1):
                new_pos = (current_pos[0], current_pos[1] - 1)
            path.append(new_pos)
            current_pos = new_pos
        return path


class MultiTargetEnv(MiniGridEnv):
    def __init__(self,
                 env_type: EnvType,
                 env_seed: int,
                 level: Level,
                 mission_space: MissionSpace,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        assert env_type not in [EnvType.GOTO, EnvType.PICKUP], "Env type can't be Goto or Pickup"
        assert level in [Level.EMPTY, Level.DIST_DIFF, Level.OPEN_DOOR, Level.BLOCKED_DOOR, Level.UNLOCK_DOOR,
                         Level.HIDDEN_KEY, Level.GO_AROUND, Level.DEATH]
        np.random.seed(env_seed)
        random.seed(env_seed)
        self.env_type = env_type
        
        # Super init
        room_size = np.random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        if max_steps is None:
            self.max_steps = 4 * room_size ** 2
        else:
            self.max_steps = max_steps
        super().__init__(mission_space = mission_space, grid_size = room_size, max_steps = self.max_steps,
                         see_through_walls = see_through_walls, render_mode = "human", agent_view_size = AGENT_VIEW_SIZE, **kwargs)
        
        # Generate random environment
        self.env_id = f"{self.env_type}-{level}-{env_seed}"
        self.objs = []  # List of (object, (x, y))
        self.walls = []  # List of (wall, x, y)
        self.doors = []  # List of (door, (x, y))
        self.keys = []  # List of (key, x, y)
        self.agent_start_pos = None
        self.agent_start_dir = None
        self.target_obj = None
        self.target_obj_pos = None
        self.auxiliary_objs = []
        self.auxiliary_objs_pos = []
        all_possible_pos = set([(x, y) for x in range(1, room_size - 1) for y in range(1, room_size - 1)])

        if level in [Level.EMPTY, Level.DIST_DIFF]:
            all_possible_pos = list(all_possible_pos)
            positions = np.random.choice(len(all_possible_pos), 2, replace = False)
            self.agent_start_pos, self.target_obj_pos = all_possible_pos[positions[0]], all_possible_pos[positions[1]]
            all_possible_pos = set(all_possible_pos)
            all_possible_pos -= set([self.agent_start_pos, self.target_obj_pos])
            self._set_target_obj(env_seed)
            self.objs = [(self.target_obj, self.target_obj_pos)]
            if level == Level.DIST_DIFF:
                num_distractors = np.random.choice(range(1, room_size - 3))
                for _ in range(num_distractors):
                    dist_obj = self.target_obj
                    while type(dist_obj) == type(self.target_obj) and dist_obj.color == self.target_obj.color:
                        dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(OBJECT_COLOR_NAMES))
                    dist_obj_pos = random.choice(list(all_possible_pos))
                    all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
            self._gen_grid(room_size, room_size)
        
        self.agent_start_dir = np.random.randint(0, 4)
            
        # Set mission
        if self.env_type == EnvType.PUT:
            self.mission = f"put the {self.auxiliary_objs[0].color} {OBJ_NAME_MAPPING[type(self.auxiliary_objs[0])]} next to the {self.target_obj.color} {OBJ_NAME_MAPPING[type(self.target_obj)]}"
        
        # Final asserts
        assert self.agent_start_pos is not None, "self.agent_start_pos is None"
        assert self.agent_start_dir is not None, "self.agent_start_dir is None"
        assert self.target_obj is not None, "self.target_obj is None"
        assert self.target_obj_pos is not None, "self.target_obj_pos is None"
        assert len(self.auxiliary_objs) != 0, "self.auxiliary_objs is empty"
        assert len(self.auxiliary_objs_pos) != 0, "self.auxiliary_objs_pos is empty"


class GotoEnv(SingleTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, GOTO_TARGET_OBJS])
        super().__init__(EnvType.GOTO, env_seed, level, mission_space, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
    
    @staticmethod
    def _gen_mission(color: str, object: str):
        return f"get to the {color} {object}"
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        ax, ay = self.agent_pos
        tx, ty = self.target_obj_pos
        if type(self.target_obj) == Goal:
            if ax == tx and ay == ty:
                reward = self._reward()
                terminated = True
        else:
            if manhattan_distance((ax, ay), (tx, ty)) == 1:
                target_obj_dir = (tx - ax == 1, ty - ay == 1, ax - tx == 1, ay - ty == 1).index(True)
                if self.agent_dir == target_obj_dir:
                    reward = self._reward()
                    terminated = True
        return obs, reward, terminated, truncated, info
    

class PickupEnv(SingleTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, PICKUP_TARGET_OBJS])
        super().__init__(EnvType.PICKUP, env_seed, level, mission_space, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
    
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


class PutEnv(MultiTargetEnv):
    def __init__(self,
                 env_seed: int,
                 level: Level,
                 max_steps: int = None,
                 see_through_walls = False,
                 **kwargs):
        mission_space = MissionSpace(mission_func = self._gen_mission,
                                     ordered_placeholders = [OBJECT_COLOR_NAMES, PICKUP_TARGET_OBJS, OBJECT_COLOR_NAMES, PICKUP_TARGET_OBJS])
        super().__init__(EnvType.PUT, env_seed, level, mission_space, max_steps = max_steps, see_through_walls = see_through_walls, **kwargs)
    
    @staticmethod
    def _gen_mission(color1: str, object1: str, color2: str, object2: str):
        return f"put the {color1} {object1} next to the {color2} {object2}"
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        u, v = self.dir_vec
        ox, oy = self.agent_pos[0] + u, self.agent_pos[1] + v
        tx, ty = self.target_obj_pos
        if self.carrying and self.carrying in self.auxiliary_objs and action == self.actions.drop:
            if abs(ox - tx) <= 1 and abs(oy - ty) <= 1:
                reward = self._reward()
                terminated = True
        return obs, reward, terminated, truncated, info
    

if __name__ == "__main__":
    seed = 759375
    env_maker = EnvMaker()
    env = env_maker.generate_env(EnvType.GOTO, env_seed = seed, level = Level.BOSS)
    manual_control = ManualControl(env, seed = seed)
    manual_control.start()
