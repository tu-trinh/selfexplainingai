from package.enums import Variant, Task, Layout
from package.envs.modifications import HeavyDoor, Bridge, FireproofShoes
from package.infrastructure.env_constants import MAX_NUM_LOCKED_DOORS
from package.infrastructure.obj_constants import (
    TANGIBLE_OBJS,
    PLAYABLE_OBJS,
    DISTRACTOR_OBJS,
)
from package.infrastructure.basic_utils import (
    flatten_list,
    debug,
    get_diagonally_adjacent_cells,
    get_adjacent_cells,
)

from minigrid.core.world_object import WorldObj, Door, Key, Goal, Wall, Lava, Box, Ball
from minigrid.wrappers import NoDeath

import numpy as np
import random
import copy
from typing import List, Tuple, Type
from abc import ABC, abstractmethod


class BaseLayout(ABC):

    def _init_layout(self):
        self.agent_dir = self.random.randint(0, 3)
        self.obstacle_thickness = 1

    def _generate_rectangular_section(
        self, obstacle_cls: bool = None
    ) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:

        on_left = 0
        on_top = 0
        horizontal_length = self.random.randint(3, self.width - 4)
        vertical_length = self.random.randint(3, self.height - 4)
        if on_top:
            row = vertical_length
            y_lb = 1
        else:
            row = self.height - 1 - vertical_length
            y_lb = row
        y_ub = y_lb + vertical_length
        if on_left:
            col = horizontal_length
            x_lb = 1
        else:
            col = self.width - 1 - horizontal_length
            x_lb = col
        x_ub = x_lb + horizontal_length

        obstacle_cells = [(x, row) for x in range(x_lb, x_ub)] + [
            (col, y) for y in range(y_lb, y_ub)
        ]
        self.obstacles = []
        self.obstacle_positions = set()
        for pos in obstacle_cells:
            self.obstacles.append(self.obstacle_cls())
            self.obstacles[-1].init_pos = pos
            self.obstacle_positions.add(pos)
        self.all_possible_pos -= set(obstacle_cells)

        self.inner_cells = set()
        self.outer_cells = set()
        t = self.obstacle_thickness
        for x, y in self.all_possible_pos:
            if x_lb + t <= x <= x_ub - t and y_lb + t <= y <= y_ub - t:
                self.inner_cells.add((x, y))
            if not (x_lb <= x <= x_ub - t and y_lb <= y <= y_ub - t):
                self.outer_cells.add((x, y))

    def _set_agent_position(self, allowed_positions: set = None) -> None:
        if allowed_positions is None:
            allowed_positions = self.all_possible_pos
        self.agent_pos = self.random.choice(list(allowed_positions))
        self.all_possible_pos -= set([self.agent_pos])

    def _set_target_position(self, allowed_positions: set = None) -> None:
        if allowed_positions is None:
            allowed_positions = self.all_possible_pos
        self.target_objects[-1].init_pos = self.random.choice(list(allowed_positions))
        self.all_possible_pos -= set([self.target_objects[-1].init_pos])

    def _make_distractor_objects(self, distractor_types: List[Type]) -> None:
        num_distractors = self.random.randint(2, min(self.width, self.height) - 3)
        self.distractor_objects = []
        for i in range(num_distractors):
            distractor_type = self.random.choice(distractor_types)
            # only one key with same color as door
            if distractor_type == Key:
                color = self.random.choice(
                    list(set(self.allowed_object_colors) - set([self.doors[0].color]))
                )
            # only one ball with same color as target
            elif distractor_type == Ball:
                color = self.random.choice(
                    list(
                        set(self.allowed_object_colors)
                        - set([self.target_objects[0].color])
                    )
                )

            if distractor_type != Wall:
                distractor = distractor_type(color=color)
            else:
                distractor = distractor_type()
            distractor.init_pos = self.random.choice(list(self.all_possible_pos))
            self.all_possible_pos -= (
                set([distractor.init_pos])
                | get_adjacent_cells(distractor.init_pos)
                | get_diagonally_adjacent_cells(distractor.init_pos)
            )
            self.distractor_objects.append(distractor)


class RoomDoorKeyLayout(BaseLayout):

    layout_name = "room_door_key"

    def _init_layout(self):

        super()._init_layout()

        self.all_possible_pos = set(
            [
                (x, y)
                for x in range(1, self.width - 1)
                for y in range(1, self.height - 1)
            ]
        )

        # generate rectangular room
        self.obstacle_cls = Wall
        self._generate_rectangular_section(Wall)

        # generate door
        self._add_door()

        # agent starts from outside room
        self._set_agent_position(self.outer_cells)

        # key must be outside room
        self.keys = []
        key_pos = self.random.choice(list(self.outer_cells))
        self.all_possible_pos -= set([key_pos]) | get_adjacent_cells(key_pos)
        self.keys.append(Key(color=self.doors[-1].color))
        self.keys[-1].init_pos = key_pos

        # target object position (before editting, there is only one target)
        self._set_target_position(self.inner_cells)

        # generator distractors
        self._make_distractor_objects([Key, Ball, Wall])

        self.objects = (
            self.doors + self.keys + self.target_objects + self.distractor_objects
        )

    def _add_door(self):
        self.doors = []
        valid_pos = []
        for pos in list(self.obstacle_positions):
            adj_cells = get_adjacent_cells(pos, ret_as_list=True)
            left, right, above, below = (
                adj_cells[1],
                adj_cells[0],
                adj_cells[2],
                adj_cells[3],
            )
            if (
                left in self.obstacle_positions and right in self.obstacle_positions
            ) or (
                above in self.obstacle_positions and below in self.obstacle_positions
            ):
                valid_pos.append(pos)
        pos = self.random.choice(valid_pos)
        # doors can't be open and locked (1, 1)
        door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
        self.doors.append(
            Door(
                is_open=door_state[0],
                is_locked=door_state[1],
                color=self.random.choice(self.allowed_object_colors),
            ),
        )
        self.doors[-1].init_pos = pos
        self.all_possible_pos -= get_adjacent_cells(pos)


class TreasureIslandLayout(BaseLayout):

    layout_name = "treasure_island"

    def _init_layout(self):

        super()._init_layout()

        self.all_possible_pos = set(
            [
                (x, y)
                for x in range(1, self.width - 1)
                for y in range(1, self.height - 1)
            ]
        )

        # generate rectangular room
        self.obstacle_cls = Lava
        self._generate_rectangular_section()

        # generate bridge
        self._add_bridge()

        # agent starts from outside island
        self._set_agent_position(self.outer_cells)

        # target object position (before editting, there is only one target)
        self._set_target_position(self.inner_cells)

        # generator distractors
        self._make_distractor_objects([Ball, Wall])

        self.objects = self.bridges + self.target_objects + self.distractor_objects

    def _add_bridge(self):
        self.bridges = []
        valid_pos = []
        for pos in list(self.obstacle_positions):
            adj_cells = get_adjacent_cells(pos, ret_as_list=True)
            left, right, above, below = (
                adj_cells[1],
                adj_cells[0],
                adj_cells[2],
                adj_cells[3],
            )
            if (
                left in self.obstacle_positions and right in self.obstacle_positions
            ) or (
                above in self.obstacle_positions and below in self.obstacle_positions
            ):
                valid_pos.append(pos)
        pos = self.random.choice(valid_pos)
        self.bridges.append(Bridge())
        self.bridges[-1].init_pos = pos
        self.all_possible_pos -= get_adjacent_cells(pos)
