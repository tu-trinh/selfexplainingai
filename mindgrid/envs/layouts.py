from __future__ import annotations

from abc import ABC
from typing import List, Tuple, Type

from minigrid.core.world_object import Ball, Key, Lava, Wall

from mindgrid.envs.editors import *
from mindgrid.envs.objects import Bridge, DoorWithDirection, Hammer
from mindgrid.envs.solvers import *
from mindgrid.infrastructure.basic_utils import (
    CustomEnum,
    get_adjacent_cells,
    get_diagonally_adjacent_cells,
)


class BaseLayout(ABC):

    def _init_layout(self):
        self.agent_init_dir = self.random.randint(0, 3)
        self.obstacle_thickness = 1

    def _reset_objects_from_state(self, state: MindGridEnvState):
        for i, o in enumerate(self.objects):
            for oo in state.objects:
                if o.type == oo.type and o.init_pos == oo.init_pos:
                    self.objects[i] = oo
                    break

    def edit(self, edits: List[Edit]):
        for e in edits:
            getattr(self, e.value)()

    def _generate_rectangular_section(
        self, obstacle_cls: bool = None
    ) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:

        assert self.obstacle_thickness == 1

        section_height = self.random.randint(3, self.width - 4)
        section_width = self.random.randint(3, self.height - 4)

        # section is always at the BOTTOM RIGHT corner
        y_min = self.height - 1 - section_height
        y_max = self.height - 1
        x_min = self.width - 1 - section_width
        x_max = self.width - 1

        self.divider_cells = set(
            [(x, y_min) for x in range(x_min, x_max)]
            + [(x_min, y) for y in range(y_min, y_max)]
        )
        self.obstacles = []
        for pos in self.divider_cells:
            self.obstacles.append(self.obstacle_cls())
            self.obstacles[-1].init_pos = pos
        self.all_possible_pos -= self.divider_cells

        self.inner_cells = set()
        self.outer_cells = set()
        for x, y in self.all_possible_pos:
            if x_min + 1 <= x <= x_max - 1 and y_min + 1 <= y <= y_max - 1:
                self.inner_cells.add((x, y))
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                self.outer_cells.add((x, y))

    def _set_agent_position(self, allowed_positions: set = None) -> None:
        if allowed_positions is None:
            allowed_positions = self.all_possible_pos
        self.agent_init_pos = self.random.choice(list(allowed_positions))
        self.all_possible_pos -= set([self.agent_init_pos])

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
    editor = RoomDoorKeyEditor
    solver = RoomDoorKeySolver

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
        self.tools = self.keys = []
        self.tool_cls = Key
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
        self.openings = self.doors = []
        self.opening_cls = DoorWithDirection
        # pick door position
        valid_pos = []
        for pos in list(self.divider_cells):
            adj_cells = get_adjacent_cells(pos, ret_as_list=True)
            left, right, above, below = (
                adj_cells[1],
                adj_cells[0],
                adj_cells[2],
                adj_cells[3],
            )
            if (left in self.divider_cells and right in self.divider_cells) or (
                above in self.divider_cells and below in self.divider_cells
            ):
                valid_pos.append(pos)
        pos = self.random.choice(valid_pos)

        # find direction vector of door
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 or j == 0) and not (i == 0 and j == 0):
                    n_pos = (pos[0] + i, pos[1] + j)
                    if n_pos in self.inner_cells:
                        dir_vec = (i, j)
        # doors can't be open and locked (1, 1)
        door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
        self.doors.append(
            self.opening_cls(
                self.random.choice(self.allowed_object_colors),
                dir_vec,
                is_open=door_state[0],
                is_locked=door_state[1],
            ),
        )
        self.doors[-1].init_pos = pos

        for i, o in enumerate(self.obstacles):
            if o.init_pos == pos:
                del self.obstacles[i]
                break

        self.all_possible_pos -= get_adjacent_cells(pos)

    def _reset_objects_from_state(self, state: MindGridEnvState):
        super()._reset_objects_from_state(state)
        self.tools = self.keys = []
        self.openings = self.doors = []
        for o in self.objects:
            if o.type == "key":
                self.keys.append(o)
            elif o.type == "door":
                self.doors.append(o)


class TreasureIslandLayout(BaseLayout):

    layout_name = "treasure_island"
    editor = TreasureIslandEditor
    solver = TreasureIslandSolver

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

        # hammer must be outside room
        self.tools = self.hammers = []
        self.tool_cls = Hammer
        hammer_pos = self.random.choice(list(self.outer_cells))
        self.all_possible_pos -= set([hammer_pos]) | get_adjacent_cells(hammer_pos)
        self.hammers.append(Hammer())
        self.hammers[-1].init_pos = hammer_pos

        # target object position (before editting, there is only one target)
        self._set_target_position(self.inner_cells)

        # generator distractors
        self._make_distractor_objects([Ball, Wall])

        self.objects = (
            self.bridges + self.hammers + self.target_objects + self.distractor_objects
        )

    def _add_bridge(self):
        self.openings = self.bridges = []
        self.opening_cls = Bridge
        valid_pos = []
        for pos in list(self.divider_cells):
            adj_cells = get_adjacent_cells(pos, ret_as_list=True)
            left, right, above, below = (
                adj_cells[1],
                adj_cells[0],
                adj_cells[2],
                adj_cells[3],
            )
            if (left in self.divider_cells and right in self.divider_cells) or (
                above in self.divider_cells and below in self.divider_cells
            ):
                valid_pos.append(pos)
        pos = self.random.choice(valid_pos)

        # find direction vector of bridge
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 or j == 0) and not (i == 0 and j == 0):
                    n_pos = (pos[0] + i, pos[1] + j)
                    if n_pos in self.inner_cells:
                        dir_vec = (i, j)

        self.bridges.append(self.opening_cls(dir_vec))
        self.bridges[-1].init_pos = pos
        self.all_possible_pos -= get_adjacent_cells(pos)

    def _reset_objects_from_state(self, state: MindGridEnvState):
        super()._reset_objects_from_state(state)
        self.tools = self.hammers = []
        self.openings = self.bridges = []
        for o in self.objects:
            if o.type == "hammer":
                self.hammers.append(o)
            elif o.type == "bridge":
                self.bridges.append(o)


class Layouts(CustomEnum):

    ROOM_DOOR_KEY = RoomDoorKeyLayout
    TREASURE_ISLAND = TreasureIslandLayout
