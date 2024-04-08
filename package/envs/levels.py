from package.enums import Variant, Task, Level
from package.envs.modifications import HeavyDoor, Bridge, FireproofShoes
from package.infrastructure.env_constants import COLOR_NAMES, MAX_NUM_LOCKED_DOORS
from package.infrastructure.obj_constants import TANGIBLE_OBJS, PLAYABLE_OBJS, DISTRACTOR_OBJS
from package.infrastructure.basic_utils import flatten_list, debug, get_diagonally_adjacent_cells, get_adjacent_cells

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Box
from minigrid.wrappers import NoDeath

import numpy as np
import random
import copy
from typing import List, Tuple
from abc import ABC, abstractmethod


class BaseLevel(ABC):
    def __init__(self):
        if Variant.ORIENTATION in self.disallowed:
            self._gen_grid(self.room_size, self.room_size)
            return
        self.all_possible_pos = set([(x, y) for x in range(1, self.room_size - 1) for y in range(1, self.room_size - 1)])
        self.agent_start_dir = np.random.randint(0, 4)
    

    @staticmethod
    def assert_successful_creation(env):
        assert env.agent_start_pos is not None, "env.agent_start_pos is None"
        assert env.agent_start_dir is not None, "env.agent_start_dir is None"
        if env.is_single_target:
            assert env.target_obj is not None, "env.target_obj is None"
            assert env.target_obj_pos is not None, "env.target_obj_pos is None"
        else:
            assert len(env.target_objs) != 0, "self.target_objs is empty"
            assert len(env.target_objs_pos) != 0, "self.target_objs_pos is empty"
    

    """
    Public Methods
    """
    @abstractmethod
    def initialize_level(self):
        pass
    

    """
    Private Methods
    """
    def _make_target_obj(self) -> None:
        if self.target_obj_type is None:
            if self.task == Task.GOTO:
                index = self.env_seed % len(PLAYABLE_OBJS)
                self.target_obj_type = PLAYABLE_OBJS[index]
            elif self.task == Task.PICKUP:
                index = self.env_seed % len(TANGIBLE_OBJS)
                self.target_obj_type = TANGIBLE_OBJS[index]
        object_colors = [cn for cn in COLOR_NAMES if cn != "grey"]
        index = self.env_seed % len(object_colors)
        if Variant.COLOR in self.disallowed:
            color = list(set(COLOR_NAMES) - set([self.disallowed[Variant.COLOR]]))[index]
        else:
            color = object_colors[index]
        if self.target_obj_type == Goal:
            self.target_obj = Goal()
            self.target_obj.color = color
        else:
            self.target_obj = self.target_obj_type(color = color)


    def _make_target_objs(self) -> None:
        if self.target_obj_types is None:
            if Variant.COLOR in self.disallowed:
                disallowed_color = self.disallowed[Variant.COLOR]
            else:
                disallowed_color = ""
            allowed_colors = [color for color in COLOR_NAMES if color != disallowed_color]
            if self.task == Task.PUT:
                idx1 = self.env_seed % len(TANGIBLE_OBJS)
                idx2 = (self.env_seed + 1) % len(PLAYABLE_OBJS)
                color1 = allowed_colors[(self.env_seed + 2) % len(allowed_colors)]
                allowed_colors.remove(color1)
                color2 = allowed_colors[(self.env_seed + 3) % len(allowed_colors)]
                pickup_obj = TANGIBLE_OBJS[idx1](color = color1)
                target_obj = PLAYABLE_OBJS[idx2]
                if target_obj == Goal:
                    target_obj = Goal()
                    target_obj.color = color2
                else:
                    target_obj = target_obj(color = color2)
                self.target_objs = [pickup_obj, target_obj]
            elif self.task == Task.COLLECT:
                collectible_obj = TANGIBLE_OBJS[self.env_seed % len(TANGIBLE_OBJS)]
                for i in range(len(self.target_objs_pos)):
                    color = allowed_colors[(self.env_seed + 4 + i) % len(allowed_colors)]
                    self.target_objs.append(collectible_obj(color = color))
            elif self.task == Task.CLUSTER:
                if len(self.target_objs) <= 5:
                    num_clusters = 2
                else:
                    num_clusters = random.choice([2, 3])
                clustered_pos = make_clusters(self.target_objs_pos, num_clusters)
                if self.env_seed % 2 == 1:  # group X-colored objects by type
                    target_color = allowed_colors[(self.env_seed + 5) % len(allowed_colors)]
                    for i in range(num_clusters):
                        this_cluster = []
                        this_cluster_type = TANGIBLE_OBJS[(self.env_seed + 6 + i) % len(TANGIBLE_OBJS)]
                        for _ in range(len(clustered_pos[i])):
                            this_cluster.append(this_cluster_type(color = target_color))
                        self.target_objs.append(this_cluster)
                else:  # group X-typed objects by color
                    target_obj = TANGIBLE_OBJS[self.env_seed % len(TANGIBLE_OBJS)]
                    for i in range(num_clusters):
                        this_cluster = []
                        this_cluster_color = allowed_colors[(self.env_seed + 6 + i) % len(allowed_colors)]
                        for _ in range(len(clustered_pos[i])):
                            random_obj = target_obj(color = this_cluster_color)
                            this_cluster.append(random_obj)
                        self.target_objs.append(this_cluster)
                self.target_objs_pos = clustered_pos
        else:
            if self.task == Task.PUT:
                color1 = allowed_colors[(self.env_seed + 3) % len(allowed_colors)]
                allowed_colors.remove(color1)
                color2 = allowed_colors[(self.env_seed + 4) % len(allowed_colors)]
                self.target_objs.append(self.target_obj_types[0](color = color1))
                self.target_objs.append(self.target_obj_types[1](color = color2))
            elif self.task == Task.COLLECT:
                color = allowed_colors[(self.env_seed + 5 + i) % len(allowed_colors)]
                self.target_objs.extend([obj(color = color) for obj in self.target_obj_types])
            elif self.task == Task.CLUSTER:
                reference = copy.deepcopy(self.target_objs_pos)
                self.target_objs_pos = []
                pos_idx = 0
                for i, obj_cluster in enumerate(self.target_obj_types):
                    this_cluster = []
                    if self.env_seed % 2 == 1:  # group X-colored objects by type
                        all_object_color = allowed_colors[(self.env_seed + 7) % len(allowed_colors)]
                        this_cluster.extend([obj(color = all_object_color) for obj in obj_cluster])
                    else:  # group X-typed objects by color
                        this_cluster_color = allowed_colors[(self.env_seed + 7 + i) % len(allowed_colors)]
                        this_cluster.extend([obj(color = this_cluster_color) for obj in obj_cluster])
                    self.target_objs.append(this_cluster)
                    self.target_objs_pos.append(reference[pos_idx : pos_idx + len(this_cluster)])
                    pos_idx += len(this_cluster)


    def _set_agent_start_position(self, allowed_positions: set = None) -> None:
        if allowed_positions:
            self.agent_start_pos = random.choice(list(allowed_positions))
            allowed_positions -= set([self.agent_start_pos])
        else:
            self.agent_start_pos = random.choice(list(self.all_possible_pos))
        self.all_possible_pos -= set([self.agent_start_pos])


    def _set_target_start_position(self, allowed_positions: set = None) -> None:
        if allowed_positions:
            self.target_obj_pos = random.choice(list(allowed_positions))
            allowed_positions -= set([self.target_obj_pos])
        else:
            self.target_obj_pos = random.choice(list(self.all_possible_pos))
        self.all_possible_pos -= set([self.target_obj_pos])
    

    def _set_targets_start_positions(self, allowed_positions: set = None, using = "all") -> None:  # FIXME: could be both inner and outer if cluster or smth
        if allowed_positions is None:
            allowed_positions = copy.deepcopy(self.all_possible_pos)
        else:
            allowed_positions = copy.deepcopy(allowed_positions)
        if self.task == Task.PUT:
            allowed_positions = list(allowed_positions)
            a, b = (0, 0), (0, 0)
            while abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1:
                positions = np.random.choice(len(allowed_positions), 2, replace = False)
                a = allowed_positions[positions[0]]
                b = allowed_positions[positions[1]]
            self.target_objs_pos = [a, b]
            allowed_positions = set(allowed_positions)
            allowed_positions -= set(self.target_objs_pos)
        elif self.task in [Task.COLLECT, Task.CLUSTER]:
            if self.task == Task.COLLECT:
                num_collectibles = random.choice(range(2, 5))
            else:
                num_collectibles = random.choice(range(3, 8))
            for _ in range(num_collectibles):
                if len(allowed_positions) > 0:
                    collectible_pos = random.choice(list(allowed_positions))
                    allowed_positions -= set([collectible_pos])
                    allowed_positions -= get_adjacent_cells(collectible_pos)
                    self.target_objs_pos.append(collectible_pos)
                else:
                    break
        if using == "all":
            self.all_possible_pos = allowed_positions
        elif using == "inner":
            self.inner_cells = allowed_positions
        elif using == "outer":
            self.outer_cells = allowed_positions


    def _set_agent_based_on_walls(self, x_lb: int, x_ub: int, y_lb: int, y_ub: int) -> None:
        # Important: upper bounds are exclusive!!!
        self.agent_start_pos = (np.random.randint(x_lb, x_ub), np.random.randint(y_lb, y_ub))
        while self.agent_start_pos not in self.all_possible_pos:
            self.agent_start_pos = (np.random.randint(x_lb, x_ub), np.random.randint(y_lb, y_ub))
        self.all_possible_pos -= set([self.agent_start_pos])


    def _set_agent_in_region(self, cell_region: set[Tuple[int, int]]) -> None:
        old_agent_pos = self.agent_start_pos
        if len(cell_region) > 0:
            new_agent_pos = random.choice(list(cell_region))
            while new_agent_pos not in self.all_possible_pos:
                new_agent_pos = random.choice(list(cell_region))
            self.all_possible_pos -= set([new_agent_pos])
            self.all_possible_pos.add(old_agent_pos)
            self.agent_start_pos = new_agent_pos
        else:  # perhaps objects took up all the space. or # FIXME: make sure always one cell left inside? and outside
            repurposed_pos = self.objs[-1][1]
            del self.objs[-1]
            matching_pos_idx = []
            for i in range(len(self.objs)):
                if self.objs[i][1] == repurposed_pos:
                    matching_pos_idx.append(i)
            matching_pos_idx.sort(reverse = True)
            for i in matching_pos_idx:
                del self.objs[i]
            self.agent_start_pos = repurposed_pos


    def _rearrange_objects(self, conflicting_positions: List[Tuple[int, int]]) -> None:
        target_objs_pos_to_remove = []
        for i in range(len(self.target_objs_pos)):
            if self.task == Task.CLUSTER:
                for j in range(len(self.target_objs_pos[i])):
                    if self.target_objs_pos[i][j] in conflicting_positions:
                        target_objs_pos_to_remove.append((i, j))
            else:
                if self.target_objs_pos[i] in conflicting_positions:
                    target_objs_pos_to_remove.append(i)
        self.all_possible_pos = list(self.all_possible_pos)
        replacement_position_idx = np.random.choice(len(self.all_possible_pos), len(target_objs_pos_to_remove), replace = False)
        replacement_positions = []
        for rp in replacement_position_idx:
            replacement_positions.append(self.all_possible_pos[rp])
        rp_idx = 0
        if self.task == Task.CLUSTER:
            for i, j in target_objs_pos_to_remove:
                self.target_objs_pos[i][j] = replacement_positions[rp_idx]
                rp_idx += 1
        else:
            for idx in target_objs_pos_to_remove:
                self.target_objs_pos[idx] = replacement_positions[rp_idx]
                rp_idx += 1
        self.all_possible_pos = set(self.all_possible_pos)
        self.all_possible_pos -= set(replacement_positions)
        if len(target_objs_pos_to_remove):
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))


    def _set_blocker_object(self):
        if self.is_single_target:
            ref_pos = self.target_obj_pos
        else:
            ref_pos = flatten_list(self.target_objs_pos)[0]
        if self.wall_orientation == "vertical":
            if ref_pos[0] > self.room_size // 2:
                blocker_obj_pos = (self.doors[0][1][0] - 1, self.doors[0][1][1])
            else:
                blocker_obj_pos = (self.doors[0][1][0] + 1, self.doors[0][1][1])
        else:
            if ref_pos[1] > self.room_size // 2:
                blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] - 1)
            else:
                blocker_obj_pos = (self.doors[0][1][0], self.doors[0][1][1] + 1)
        self.all_possible_pos -= set([blocker_obj_pos])

        if self.is_single_target:
            blocker_obj = self.target_obj
        else:
            blocker_obj = flatten_list(self.target_objs)[0]
        disallowed_blocker_obj_config = set([(type(blocker_obj), blocker_obj.color)])
        if Variant.OBJECTS in self.disallowed:
            disallowed_blocker_obj, disallowed_blocker_color = self.disallowed[Variant.OBJECTS][0][-1]
            disallowed_blocker_obj_config.add((disallowed_blocker_obj, disallowed_blocker_color))
        for existing_obj, _ in self.objs:
            disallowed_blocker_obj_config.add((type(existing_obj), existing_obj.color))
        while (type(blocker_obj), blocker_obj.color) in disallowed_blocker_obj_config:
            blocker_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(COLOR_NAMES))
        self.objs.append((blocker_obj, blocker_obj_pos))
        self.blocker_obj = blocker_obj


    def _build_walls_and_doors_based_on_target(self) -> Tuple[int, int, int, int]:
        if self.is_single_target:
            ref_pos = self.target_obj_pos
        else:
            temp = flatten_list(self.target_objs_pos)
            ref_pos = temp[0]
        # Figure out wall orientation, length, and positioning
        self.wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
        if self.wall_orientation == "vertical":
            if self.is_single_target:
                if ref_pos[0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([ref_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([ref_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
            else:  # FIXME: (check FIX NOTES; this probably is only best fit for PUT task) (make sure to do for the go round one too)
                first_obj_x = self.target_objs_pos[0][0]
                second_obj_x = self.target_objs_pos[1][0]
                wall_col = np.random.choice(list(set(range(first_obj_x + 1, second_obj_x))))
                other_side_x_lb, other_side_x_ub = 1, wall_col
                other_side_y_lb, other_side_y_ub = 1, self.room_size - 1
            self.walls = [(Wall(), (wall_col, y)) for y in range(1, self.room_size - 1)]
        elif self.wall_orientation == "horizontal":
            if self.is_single_target:
                if ref_pos[1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([ref_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([ref_pos[1]])))
                    other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
            else:
                first_obj_y = self.target_objs_pos[0][1]
                second_obj_y = self.target_objs_pos[1][1]
                wall_row = np.random.choice(list(set(range(first_obj_y + 1, second_obj_y))))
                other_side_x_lb, other_side_x_ub = 1, self.room_size - 1
                other_side_y_lb, other_side_y_ub = 1, wall_row
            self.walls = [(Wall(), (x, wall_row)) for x in range(1, self.room_size - 1)]
        wall_positions = [wall[1] for wall in self.walls]
        self.all_possible_pos -= set(wall_positions)

        # Possible object rearrangement
        if not self.is_single_target:
            self._rearrange_objects(wall_positions)

        # Establish doors
        self.doors = [(Door(is_locked = self.level in [Level.UNLOCK_DOOR, Level.HIDDEN_KEY], color = random.choice(COLOR_NAMES)), random.choice(wall_positions))]

        # Return for future calculations
        return other_side_x_lb, other_side_x_ub, other_side_y_lb, other_side_y_ub
    

    def _build_partial_walls_and_doors_based_on_target(self) -> Tuple[int, int, int, int]:
        if self.is_single_target:
            ref_pos = self.target_obj_pos
        else:
            ref_pos = flatten_list(self.target_objs_pos)[0]
        
        # Figure out wall orientation, length, and positioning
        self.wall_orientation = "vertical" if np.random.random() > 0.5 else "horizontal"
        if self.wall_orientation == "vertical":
            if self.is_single_target:
                if ref_pos[0] > self.room_size // 2:
                    wall_col = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([ref_pos[0]])))
                    other_side_x_lb, other_side_x_ub = 1, wall_col
                else:
                    wall_col = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([ref_pos[0]])))
                    other_side_x_lb, other_side_x_ub = wall_col + 1, self.room_size - 1
            else:
                first_obj_x = self.target_objs_pos[0][0]
                second_obj_x = self.target_objs_pos[1][0]
                wall_col = np.random.choice(list(set(range(first_obj_x + 1, second_obj_x))))
                other_side_x_lb, other_side_x_ub = 1, wall_col
            if ref_pos[1] > self.room_size // 2:
                wall_head = random.choice(range(2, ref_pos[1]))
                wall_tail = self.room_size - 1
            else:
                wall_head = 1
                wall_tail = random.choice(range(ref_pos[1] + 1, self.room_size - 1))
            self.walls = [(Wall(), (wall_col, y)) for y in range(wall_head, wall_tail)]

            return_tuple = (other_side_x_lb, other_side_x_ub, wall_head, wall_tail)
        
        elif self.wall_orientation == "horizontal":
            if self.is_single_target:
                if ref_pos[1] > self.room_size // 2:
                    wall_row = np.random.choice(list(set(range(2, self.room_size // 2 + 1)) - set([ref_pos[1]])))
                    other_side_y_lb, other_side_y_ub = 1, wall_row
                else:
                    wall_row = np.random.choice(list(set(range(self.room_size // 2 + 1, self.room_size - 2)) - set([ref_pos[1]])))
                    other_side_y_lb, other_side_y_ub = wall_row + 1, self.room_size - 1
            else:
                first_obj_y = self.target_objs_pos[0][1]
                second_obj_y = self.target_objs_pos[1][1]
                wall_row = np.random.choice(list(set(range(first_obj_y + 1, second_obj_y))))
                other_side_y_lb, other_side_y_ub = 1, wall_row
            if ref_pos[0] > self.room_size // 2:
                wall_head = random.choice(range(2, ref_pos[0]))
                wall_tail = self.room_size - 1
            else:
                wall_head = 1
                wall_tail = random.choice(range(ref_pos[0] + 1, self.room_size - 1))
            self.walls = [(Wall(), (x, wall_row)) for x in range(wall_head, wall_tail)]

            return_tuple = (wall_head, wall_tail, other_side_y_lb, other_side_y_ub)

        wall_positions = [wall[1] for wall in self.walls]
        self.all_possible_pos -= set(wall_positions)

        # Establish doors
        self.doors = [(Door(color = random.choice(COLOR_NAMES)), random.choice(wall_positions))]

        # Return for future calculations
        return return_tuple


    def _set_key_for_door(self, x_lb: int, x_ub: int, y_lb: int, y_ub: int, hide_key: bool) -> None:
        key = Key(color = self.doors[0][0].color)
        key_pos = (np.random.randint(x_lb, x_ub), np.random.randint(y_lb, y_ub))
        self.all_possible_pos -= set([key_pos])
        if hide_key:
            disallowed_box_colors = set()
            if self.is_single_target:
                if type(self.target_obj) == Box:
                    disallowed_box_colors.add(self.target_obj.color)
            else:
                for to in flatten_list(self.target_objs):
                    if type(to) == Box:
                        disallowed_box_colors.add(to.color)
            box = Box(color = random.choice(list(set(COLOR_NAMES) - disallowed_box_colors)))
            box.contains = key
            self.objs.append((box, key_pos))
        else:
            self.keys.append((key, key_pos))
    

    def _gen_path_to_target(self) -> set[Tuple[int, int]]:
        total_path = set()
        if self.is_single_target:
            targets = [self.target_obj_pos]
        else:
            targets = self.target_objs_pos
        for target_pos in flatten_list(targets):
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


    def _get_cells_in_partition(self, start_x: int, end_x: int, start_y: int, end_y: int) -> List[Tuple[int, int]]:
        return [(x, y) for x in range(start_x, end_x + 1) for y in range(start_y, end_y + 1)]
    
    
    def _generate_walls_for_partition(self, start_x: int, end_x: int, start_y: int, end_y: int, min_subroom_size: int = 2) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
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
    
    
    def _gen_multiple_rooms(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[set[Tuple[int, int]]], List[bool]]:
        walls = []
        doors = []
        partitions = [(1, self.room_size - 2, 1, self.room_size - 2)]  # Initial partition covering the whole room
        partition_cells = []
        partition_entryways = []
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
            this_partition_cells = set()
            partition_has_door = False
            for cell in cells:
                if cell not in walls:
                    if cell in doors:
                        partition_has_door = True
                    else:
                        this_partition_cells.add(cell)
            partition_entryways.append(partition_has_door)
            partition_cells.append(this_partition_cells)
        return walls, doors, partition_cells, partition_entryways


    def _generate_rectangular_section(self, is_walls: bool) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
        on_left = np.random.random() < 0.5
        on_top = np.random.random() < 0.5
        horizontal_length = np.random.randint(3, self.room_size - 3)
        vertical_length = np.random.randint(3, self.room_size - 3)
        if on_top:
            row = vertical_length
            y_lb = 1
        else:
            row = self.room_size - 1 - vertical_length
            y_lb = row
        y_ub = y_lb + vertical_length
        if on_left:
            col = horizontal_length
            x_lb = 1
        else:
            col = self.room_size - 1 - horizontal_length
            x_lb = col
        x_ub = x_lb + horizontal_length
        
        section = [(x, row) for x in range(x_lb, x_ub)]
        section += [(col, y) for y in range(y_lb, y_ub)]
        if is_walls:
            self.walls.extend([(Wall(), pos) for pos in section])
        else:
            self.objs.extend([(Lava(), pos) for pos in section])
        self.all_possible_pos -= set(section)

        inner_cells = set()
        outer_cells = set()
        for x, y in self.all_possible_pos:
            if x_lb <= x < x_ub and y_lb <= y < y_ub:
                inner_cells.add((x, y))
            else:
                outer_cells.add((x, y))
        return inner_cells, outer_cells


    def _make_distractor_objects(self, additional_allowable_pos = None) -> None:
        for i in range(self.num_distractors):
            if self.is_single_target:
                dist_obj = self.target_obj
            else:
                temp = flatten_list(self.target_objs)
                dist_obj = temp[0]
            while (type(dist_obj), dist_obj.color) in self.disallowed_obj_config:
                dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(COLOR_NAMES))
            if Variant.OBJECTS in self.disallowed and self.required_obj_positions[i] in self.all_possible_pos and (additional_allowable_pos is None or (self.required_obj_positions[i] in additional_allowable_pos)):
                dist_obj_pos = self.required_obj_positions[i]
            else:
                dist_obj_pos = random.choice(list(self.all_possible_pos))
            self.all_possible_pos -= set([dist_obj_pos])
            self.all_possible_pos -= get_adjacent_cells(dist_obj_pos)
            self.all_possible_pos -= get_diagonally_adjacent_cells(dist_obj_pos)
            self.objs.append((dist_obj, dist_obj_pos))
            self.disallowed_obj_config.add((type(dist_obj), dist_obj.color))


    def _determine_num_distractors(self, lb: int, ub: int) -> None:
        if Variant.NUM_OBJECTS in self.disallowed:
            self.num_distractors = random.choice(list(set(range(lb, ub)) - set([self.disallowed[Variant.NUM_OBJECTS]])))
        elif Variant.OBJECTS in self.disallowed:
            self.required_obj_positions = self.disallowed[Variant.OBJECTS][1]
            self.num_distractors = len(required_obj_positions)
        else:
            self.num_distractors = np.random.choice(range(lb, ub))


    def _determine_num_rooms(self) -> None:
        rooms_range = list(range(2, 5 if self.room_size <= 9 else 7))
        if Variant.NUM_ROOMS in self.disallowed:
            self.num_rooms = random.choice(list(set(rooms_range) - set([self.disallowed[Variant.NUM_ROOMS]])))
        else:
            self.num_rooms = random.choice(rooms_range)


class EmptyLevel(BaseLevel):
    def initialize_level(self):
        # Set agent and target objects
        self._set_agent_start_position()
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))


class DeathLevel(BaseLevel):
    def initialize_level(self):
        # Set agent and target objects
        self._set_agent_start_position()
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))
        
        # Sprinkle lava around
        path_to_target = self._gen_path_to_target()
        self.all_possible_pos -= path_to_target
        if self.is_single_target:
            num_lavas = random.choice(range(int(0.25 * (self.room_size - 2)**2), int(0.4 * (self.room_size - 2)**2)))
        else:
            num_lavas = min(len(self.all_possible_pos) - 1, random.choice(range(int(0.25 * (self.room_size - 2)**2), int(0.4 * (self.room_size - 2)**2))))
        self.all_possible_pos = list(self.all_possible_pos)
        lava_positions = np.random.choice(len(self.all_possible_pos), num_lavas, replace = False)
        for p in lava_positions:
            self.objs.append((Lava(), self.all_possible_pos[p]))
        for p in sorted(lava_positions, reverse = True):
            del self.all_possible_pos[p]


class DistractorsLevel(BaseLevel):
    def initialize_level(self):
        # Set agent and target object
        self._set_agent_start_position()
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set distractor objects
        if self.is_single_target:
            self.disallowed_obj_config = set([(type(self.target_obj), self.target_obj.color)])
        else:
            self.disallowed_obj_config = set([(type(to), to.color) for to in flatten_list(self.target_objs)])
        for existing_obj, _ in self.objs:
            self.disallowed_obj_config.add((type(existing_obj), existing_obj.color))
        if Variant.OBJECTS in self.disallowed:
            self.disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
        self._determine_num_distractors(1, self.room_size - 3)
        self._make_distractor_objects()


class OpenDoorLevel(BaseLevel):
    def initialize_level(self):
        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set walls, doors, and agent
        x_lb, x_ub, y_lb, y_ub = self._build_walls_and_doors_based_on_target()
        self._set_agent_based_on_walls(x_lb, x_ub, y_lb, y_ub)


class BlockedDoorLevel(BaseLevel):
    def initialize_level(self):
        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set walls, door, object that blocks the door, and aget
        x_lb, x_ub, y_lb, y_ub = self._build_walls_and_doors_based_on_target()
        self._set_blocker_object()
        self._set_agent_based_on_walls(x_lb, x_ub, y_lb, y_ub)


class UnlockDoorLevel(BaseLevel):
    def initialize_level(self):
        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set walls, door, key, and agent
        x_lb, x_ub, y_lb, y_ub = self._build_walls_and_doors_based_on_target()
        self._set_key_for_door(x_lb, x_ub, y_lb, y_ub, False)
        self._set_agent_based_on_walls(x_lb, x_ub, y_lb, y_ub)


class HiddenKeyLevel(BaseLevel):
    def initialize_level(self):
        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set walls, door, key, and agent
        x_lb, x_ub, y_lb, y_ub = self._build_walls_and_doors_based_on_target()
        self._set_key_for_door(x_lb, x_ub, y_lb, y_ub, True)
        self._set_agent_based_on_walls(x_lb, x_ub, y_lb, y_ub)


class GoAroundLevel(BaseLevel):
    def initialize_level(self):
        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Set walls and agent
        x_lb, x_ub, y_lb, y_ub = self._build_partial_walls_and_doors_based_on_target()
        self._set_agent_based_on_walls(x_lb, x_ub, y_lb, y_ub)


class MultipleRoomsLevel(BaseLevel):
    def initialize_level(self):
        # Determine number of rooms and then make rooms and doors
        self._determine_num_rooms()
        room_walls, room_doors, room_cells, door_markers = self._gen_multiple_rooms()
        self.walls.extend([(Wall(), pos) for pos in room_walls + room_doors])
        available_door_colors = copy.deepcopy(COLOR_NAMES)
        for pos in room_doors:
            chosen_color = random.choice(available_door_colors)
            self.doors.append((Door(is_locked = False, color = chosen_color), pos))
            available_door_colors.remove(chosen_color)
        self.all_possible_pos -= set(room_walls)
        self.all_possible_pos -= set(room_doors)

        # Place agent in its own room
        self._set_agent_start_position()
        for i in range(len(room_cells)):
            if self.agent_start_pos in room_cells[i]:
                self.all_possible_pos -= room_cells[i]
                break

        # Set target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))


class RoomDoorKeyLevel(BaseLevel):
    def initialize_level(self):
        # Generate rectangular room and door
        self.inner_cells, self.outer_cells = self._generate_rectangular_section(True)
        valid_door_pos = []
        wall_positions = [pos for _, pos in self.walls]
        for pos in wall_positions:
            adj_cells = get_adjacent_cells(pos, ret_as_list = True)
            left, right = adj_cells[1], adj_cells[0]
            above, below = adj_cells[2], adj_cells[3]
            if (left in wall_positions and right in wall_positions) or (above in wall_positions and below in wall_positions):
                valid_door_pos.append(pos)
        door_pos = random.choice(valid_door_pos)
        self.doors.append((Door(is_locked = True, color = random.choice(COLOR_NAMES)), door_pos))
        self.all_possible_pos -= get_adjacent_cells(door_pos)  # FIXME: sort of a temporary fix because there's no easy way to tell if door in this environment is blocked but it still works tbh

        # Place agent and key outside of the room, target inside the room
        self._set_agent_start_position(self.outer_cells)
        key_pos = random.choice(list(self.outer_cells))
        self.all_possible_pos -= set([key_pos])
        self.all_possible_pos -= get_adjacent_cells(key_pos)
        self.all_possible_pos -= get_diagonally_adjacent_cells(key_pos)
        self.keys.append((Key(color = self.doors[0][0].color), key_pos))
        if self.is_single_target:
            self._set_target_start_position(self.inner_cells)
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions(allowed_positions = self.inner_cells, using = "inner")
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Generate objects inside room
        if self.is_single_target:
            self.disallowed_obj_config = set([(type(self.target_obj), self.target_obj.color)])
        else:
            self.disallowed_obj_config = set([(type(to), to.color) for to in flatten_list(self.target_objs)])
        if Variant.OBJECTS in self.disallowed:
            self.disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
        for existing_obj, _ in self.objs:
            self.disallowed_obj_config.add((type(existing_obj), existing_obj.color))
        self._determine_num_distractors(1, self.room_size - 3)
        self._make_distractor_objects(self.inner_cells)
    

    def add_opening_to_wall(self, is_door = True):
        available_openings = set([pos for _, pos in self.walls]) - set([pos for _, pos in self.doors])
        opening_pos = random.choice(list(available_openings))
        if is_door:
            self.doors.append((Door(is_locked = False, color = random.choice(COLOR_NAMES)), opening_pos))
        else:
            for i in range(len(self.walls)):
                if self.walls[i][1] == opening_pos:
                    to_remove = i
                    break
            del self.walls[to_remove]
        self._gen_grid(self.room_size, self.room_size)
    

    def block_door(self):
        if len(self.doors) == 0:
            warnings.warn("Cannot block door in environment without doors")
            return
        wall_positions = [pos for _, pos in self.walls]
        obj_positions = [pos for _, pos in self.objs]
        key_positions = [pos for _, pos in self.keys]
        used_positions = wall_positions + obj_positions + key_positions
        failed = False
        for _, (x, y) in self.doors:
            blocker_pos = None
            adj_cells = get_adjacent_cells((x, y))
            for adj_cell in adj_cells:
                if adj_cell not in used_positions and adj_cell in self.outer_cells and adj_cell in self.all_possible_pos:
                    blocker_pos = adj_cell
                    break
            if not blocker_pos:
                failed = True
            else:
                self.all_possible_pos -= set([blocker_pos])
                if self.is_single_target:
                    blocker_obj = self.target_obj
                else:
                    blocker_obj = flatten_list(self.target_objs)[0]
                disallowed_blocker_obj_config = set([(type(blocker_obj), blocker_obj.color)])
                while (type(blocker_obj), blocker_obj.color) in disallowed_blocker_obj_config:
                    blocker_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(COLOR_NAMES))
                self.objs.append((blocker_obj, blocker_pos))
        self._gen_grid(self.room_size, self.room_size)
        if failed:
            warnings.warn("Failed to block at least one door")


    def put_agent_in_room(self):
        self._set_agent_in_region(self.inner_cells)
        self._gen_grid(self.room_size, self.room_size)


class TreasureIslandLevel(BaseLevel):
    def initialize_level(self):
        # Generate section blocked off by lava
        self.inner_cells, self.outer_cells = self._generate_rectangular_section(False)
        valid_bridge_pos = []
        lava_positions = [pos for obj, pos in self.objs if type(obj) == Lava]
        for pos in lava_positions:
            adj_cells = get_adjacent_cells(pos, ret_as_list = True)
            left, right = adj_cells[1], adj_cells[0]
            above, below = adj_cells[2], adj_cells[3]
            if (left in lava_positions and right in lava_positions) or (above in lava_positions and below in lava_positions):
                valid_bridge_pos.append(pos)
        bridge_pos = random.choice(valid_bridge_pos)
        self.objs.append((Bridge(), bridge_pos))
        self.all_possible_pos -= get_adjacent_cells(bridge_pos)  # FIXME: sort of a temporary fix because there's no easy way to tell if door in this environment is blocked but it still works tbh

        # Place agent outside of the section, target inside the section
        self._set_agent_start_position(self.outer_cells)
        if self.is_single_target:
            self._set_target_start_position(self.inner_cells)
            self._make_target_obj()
            self.objs.extend([(self.target_obj, self.target_obj_pos)])
        else:
            self._set_targets_start_positions(allowed_positions = self.inner_cells, using = "inner")
            self._make_target_objs()
            self.objs.extend(list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos))))

        # Generate objects inside room
        self._determine_num_distractors(1, self.room_size - 3)
        if self.is_single_target:
            self.disallowed_obj_config = set([(type(self.target_obj), self.target_obj.color)])
        else:
            self.disallowed_obj_config = set([(type(to), to.color) for to in flatten_list(self.target_objs)])
        if Variant.OBJECTS in self.disallowed:
            self.disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
        for existing_obj, _ in self.objs:
            self.disallowed_obj_config.add((type(existing_obj), existing_obj.color))
        self._make_distractor_objects(self.inner_cells)
    

    def add_bridge(self):
        lava_positions = [pos for obj, pos in self.objs if type(obj) == Lava]
        bridge_pos = random.choice(lava_positions)
        for i in range(len(self.objs)):
            if self.objs[i][1] == bridge_pos:
                to_remove = i
                break
        del self.objs[to_remove]
        self.objs.append((Bridge(), bridge_pos))
        self._gen_grid(self.room_size, self.room_size)

    
    def make_lava_safe(self, lava_cost = 0):  # FIXME: which one?
        # self = NoDeath(self, no_death_types = ("lava",), death_cost = lava_cost)
        for i in range(len(self.objs)):
            if type(self.objs[i][0]) == Lava:
                replacement_lava = Goal()
                replacement_lava.color = "red"
                self.objs[i] = (replacement_lava, self.objs[i][1])
        self._gen_grid(self.room_size, self.room_size)

    
    def add_fireproof_shoes(self):
        shoe_pos = random.choice(list(self.outer_cells))
        while shoe_pos not in self.all_possible_pos:
            shoe_pos = random.choice(list(self.outer_cells))
        self.all_possible_pos -= set([shoe_pos])
        self.objs.append((FireproofShoes(), shoe_pos))
        self._gen_grid(self.room_size, self.room_size)
    

    def put_agent_on_island(self):
        self._set_agent_in_region(self.inner_cells)
        self._gen_grid(self.room_size, self.room_size)


class BossLevel(BaseLevel):
    def initialize_level(self):
        # Handling MULT_ROOMS characteristics
        self._determine_num_rooms()
        room_walls, room_doors, room_cells, door_markers = self._gen_multiple_rooms()
        self.walls.extend([(Wall(), pos) for pos in room_walls + room_doors])
        self.all_possible_pos -= set(room_walls)
        self.all_possible_pos -= set(room_doors)

        # Handle UNLOCK_DOOR characteristics
        necessary_key_colors = []
        locked_doors = 0
        available_door_colors = copy.deepcopy(COLOR_NAMES)
        for room_door_pos in room_doors:
            is_locked = random.choice([True, False])
            chosen_color = random.choice(COLOR_NAMES)
            available_door_colors.remove(chosen_color)
            door = Door(is_locked = is_locked and locked_doors < MAX_NUM_LOCKED_DOORS, color = chosen_color)
            if is_locked:
                necessary_key_colors.append(chosen_color)
                locked_doors += 1
            self.doors.append((door, room_door_pos))
        self._set_agent_start_position()
        for i in range(len(room_cells)):
            if self.agent_start_pos in room_cells[i]:
                room_cells[i].remove(self.agent_start_pos)
                for key_color in necessary_key_colors:
                    key_pos = random.choice(list(room_cells[i]))
                    room_cells[i].remove(key_pos)
                    self.keys.append((Key(color = key_color), key_pos))
                self.all_possible_pos -= room_cells[i]
                agent_cell_idx = i
                break
        del room_cells[agent_cell_idx]

        # Set the target objects
        if self.is_single_target:
            self._set_target_start_position()
            self._make_target_obj()
            self.objs = [(self.target_obj, self.target_obj_pos)]
        else:
            self._set_targets_start_positions()
            self._make_target_objs()
            self.objs = list(zip(flatten_list(self.target_objs), flatten_list(self.target_objs_pos)))

        # Handle DIST and DEATH characteristics
        smallest_room = len(min(room_cells, key = len))
        self._determine_num_distractors(1, smallest_room)
        if self.is_single_target:
            self.disallowed_obj_config = set([(type(self.target_obj), self.target_obj.color)])
        else:
            self.disallowed_obj_config = set([(type(obj), obj.color) for obj in flatten_list(self.target_objs)])
        if Variant.OBJECTS in self.disallowed:
            self.disallowed_obj_config.update(self.disallowed[Variant.OBJECTS][0])
        for existing_obj, _ in self.objs:
            self.disallowed_obj_config.add((type(existing_obj), existing_obj.color))
        for i in range(self.num_distractors):
            actually_lava = random.choice([True, False])
            if actually_lava:
                lava_pos = None
                if Variant.OBJECTS in self.disallowed and self.required_obj_positions[i] in self.all_possible_pos:
                    lava_pos = self.required_obj_positions[i]
                else:
                    if len(self.all_possible_pos) > 0:
                        lava_pos = random.choice(list(self.all_possible_pos))
                if lava_pos:
                    self.all_possible_pos -= set([lava_pos])
                    self.objs.append((Lava(), lava_pos))
            else:
                dist_obj_pos = None
                if self.is_single_target:
                    dist_obj = self.target_obj
                else:
                    dist_obj = flatten_list(self.target_objs)[0]
                while (type(dist_obj), dist_obj.color) in self.disallowed_obj_config:
                    dist_obj = random.choice(DISTRACTOR_OBJS)(color = random.choice(COLOR_NAMES))
                if Variant.OBJECTS in self.disallowed and self.required_obj_positions[i] in self.all_possible_pos:
                    dist_obj_pos = self.required_obj_positions[i]
                else:
                    if len(self.all_possible_pos) > 0:
                        dist_obj_pos = random.choice(list(self.all_possible_pos))
                if dist_obj_pos:
                    self.all_possible_pos -= set([dist_obj_pos])
                    self.objs.append((dist_obj, dist_obj_pos))
