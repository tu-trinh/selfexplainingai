from __future__ import annotations

from abc import ABC
from copy import deepcopy as dc
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
from mindgrid.infrastructure.env_utils import are_objects_equal


class ObjectManager:

    def __init__(self):
        self.store = {}
        self.store["all"] = []

    def add(self, name, obj, init_pos=None):
        if name in self.store:
            self.store[name].append(obj)
        else:
            self.store[name] = [obj]
        if init_pos is not None:
            obj.init_pos = init_pos
        self.store["all"].append(obj)

    def remove(self, name, obj):
        self.store[name].remove(obj)
        self.store["all"].remove(obj)

    def types(self):
        ret = list(self.store.keys())
        ret.remove("all")
        return ret

    def __eq__(self, other):
        if list(self.store.keys()) != list(other.store.keys()):
            return False
        for k in self.store.keys():
            if len(self.store[k]) != len(other.store[k]):
                return False
            for o, oo in zip(self.store[k], other.store[k]):
                if not are_objects_equal(o, oo):
                    return False
        return True

    def __iter__(self):
        return iter(self.store["all"])

    def __getitem__(self, name):
        return self.store[name]

    def __contains__(self, name):
        return name in self.store

    def __len__(self):
        return len(self.store["all"])


class BaseLayout(ABC):

    @property
    def targets(self):
        return self.objects["target"]

    @property
    def init_targets(self):
        return self.init_objects["target"]

    @property
    def distractors(self):
        return self.objects["distractor"]

    @property
    def init_distractors(self):
        return self.init_objects["distractor"]

    def _init_layout(self):
        self.all_possible_pos = set(
            [
                (x, y)
                for x in range(1, self.width - 1)
                for y in range(1, self.height - 1)
            ]
        )

        # generate rectangular section
        self._generate_rectangular_section(self.obstacle_cls)

        # agent starts from outside section
        self._put_agent()

        # object manager
        self.init_objects = ObjectManager()

        # add target inside the section
        self._add_target()

    def _reset_objects(self, state: MindGridEnvState = None):
        if state is None:
            self.objects = dc(self.init_objects)
        else:
            self.objects = state.objects

    def _generate_rectangular_section(
        self, obstacle_cls: bool = None
    ) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:

        self.obstacle_thickness = 1

        section_height = self.random.randint(4, self.width - 4)
        section_width = self.random.randint(4, self.height - 4)

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

    def _add_target(self):
        allowed_positions = dc(self.inner_cells)
        # don't put target next to wall
        for c in self.divider_cells:
            for cc in get_adjacent_cells(c):
                allowed_positions.discard(cc)
        init_pos = self.random.choice(list(allowed_positions))
        self.init_objects.add(
            "target", self.target_cls(color=self.target_color), init_pos=init_pos
        )
        self.all_possible_pos -= set([init_pos]) | get_adjacent_cells(init_pos)

    def _add_distractors(self, distractor_types: List[Type]) -> None:
        num_distractors = self.random.randint(2, min(self.width, self.height) // 2)
        for i in range(num_distractors):
            distractor_type = self.random.choice(distractor_types)
            # only one key with same color as door
            if distractor_type == Key:
                color = self.random.choice(
                    list(
                        set(self.allowed_object_colors)
                        - set([self.init_doors[0].color])
                    )
                )
            # only one ball with same color as target
            elif distractor_type == Ball:
                color = self.random.choice(
                    list(
                        set(self.allowed_object_colors)
                        - set([self.init_targets[0].color])
                    )
                )

            if distractor_type != Wall:
                distractor = distractor_type(color=color)
            else:
                distractor = distractor_type()

            init_pos = self.random.choice(list(self.all_possible_pos))
            self.all_possible_pos -= (
                set([init_pos])
                | get_adjacent_cells(init_pos)
                | get_diagonally_adjacent_cells(init_pos)
            )
            self.init_objects.add("distractor", distractor, init_pos=init_pos)

    def _put_agent(self):
        self.init_agent_pos = self.random.choice(list(self.outer_cells))
        self.all_possible_pos -= set([self.init_agent_pos])
        self.init_agent_dir = self.random.randint(0, 3)


class RoomDoorKeyLayout(BaseLayout):

    layout_name = "room_door_key"
    opening_name = "door"
    tool_name = "key"
    editor = RoomDoorKeyEditor
    solver = RoomDoorKeySolver

    @property
    def opening_name(self):
        return "door"

    @property
    def tool_name(self):
        return "key"

    @property
    def doors(self):
        return self.objects["door"]

    @property
    def keys(self):
        return self.objects["key"]

    @property
    def openings(self):
        return self.doors

    @property
    def tools(self):
        return self.keys

    @property
    def init_doors(self):
        return self.init_objects["door"]

    @property
    def init_keys(self):
        return self.init_objects["key"]

    @property
    def init_openings(self):
        return self.init_doors

    @property
    def init_tools(self):
        return self.init_keys

    @property
    def opening_cls(self):
        return DoorWithDirection

    @property
    def tool_cls(self):
        return Key

    def _init_layout(self):
        self.obstacle_cls = Wall
        super()._init_layout()
        self._add_door()
        self._add_key()
        self._add_distractors([Key, Ball, Wall])

    def _add_door(self):
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
        init_pos = self.random.choice(valid_pos)

        # find direction vector of door
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 or j == 0) and not (i == 0 and j == 0):
                    n_pos = (init_pos[0] + i, init_pos[1] + j)
                    if n_pos in self.inner_cells:
                        dir_vec = (i, j)
        # doors can't be open and locked (1, 1)
        door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
        self.init_objects.add(
            "door",
            self.opening_cls(
                self.random.choice(self.allowed_object_colors),
                dir_vec,
                is_open=door_state[0],
                is_locked=door_state[1],
            ),
            init_pos=init_pos,
        )

        # delete obstacles at the door's location
        for i, o in enumerate(self.obstacles):
            if o.init_pos == init_pos:
                del self.obstacles[i]
                break

        self.all_possible_pos -= get_adjacent_cells(init_pos)

    def _add_key(self):
        # key must be outside room
        init_pos = self.random.choice(list(self.outer_cells & self.all_possible_pos))
        self.all_possible_pos -= set([init_pos]) | get_adjacent_cells(init_pos)
        self.init_objects.add(
            "key", Key(color=self.init_doors[-1].color), init_pos=init_pos
        )


class TreasureIslandLayout(BaseLayout):

    layout_name = "treasure_island"
    opening_name = "bridge"
    tool_name = "hammer"
    editor = RoomDoorKeyEditor
    solver = RoomDoorKeySolver

    editor = TreasureIslandEditor
    solver = TreasureIslandSolver

    @property
    def opening_name(self):
        return "bridge"

    @property
    def tool_name(self):
        return "hammer"

    @property
    def bridges(self):
        return self.objects["bridge"]

    @property
    def hammers(self):
        return self.objects["hammer"]

    @property
    def openings(self):
        return self.bridges

    @property
    def tools(self):
        return self.hammers

    @property
    def init_bridges(self):
        return self.init_objects["bridge"]

    @property
    def init_hammers(self):
        return self.init_objects["hammer"]

    @property
    def init_openings(self):
        return self.init_bridges

    @property
    def init_tools(self):
        return self.init_hammers

    @property
    def opening_cls(self):
        return Bridge

    @property
    def tool_cls(self):
        return Hammer

    def _init_layout(self):
        self.obstacle_cls = Lava
        super()._init_layout()
        self._add_bridge()
        self._add_hammer()
        self._add_distractors([Ball, Wall])

    def _add_bridge(self):
        valid_pos = []
        for pos in list(set(self.divider_cells)):
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
        init_pos = self.random.choice(valid_pos)

        # find direction vector of bridge
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 or j == 0) and not (i == 0 and j == 0):
                    n_pos = (init_pos[0] + i, init_pos[1] + j)
                    if n_pos in self.inner_cells:
                        dir_vec = (i, j)

        self.init_objects.add(
            "bridge",
            self.opening_cls(dir_vec, is_intact=self.random.choice([0, 1])),
            init_pos=init_pos,
        )

        # delete obstacles at the bridge's location
        for i, o in enumerate(self.obstacles):
            if o.init_pos == init_pos:
                del self.obstacles[i]
                break

        self.all_possible_pos -= get_adjacent_cells(pos)

    def _add_hammer(self):
        # hammer must be outside section
        init_pos = self.random.choice(list(self.outer_cells & self.all_possible_pos))
        self.all_possible_pos -= set([init_pos]) | get_adjacent_cells(init_pos)
        self.init_objects.add("hammer", Hammer(), init_pos=init_pos)


class Layouts(CustomEnum):

    room_door_key = RoomDoorKeyLayout
    treasure_island = TreasureIslandLayout
