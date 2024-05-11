from __future__ import annotations

from typing import List

from minigrid.core.world_object import Ball, Box, WorldObj

from mindgrid.envs.objects import Bridge, FireproofShoes, Passage, SafeLava
from mindgrid.infrastructure.basic_utils import CustomEnum, get_adjacent_cells


class BaseEditor:

    def edit(self, edits: List[Edit]):
        for e in edits:
            getattr(self, e.value)()

    def none(self):
        pass

    def double_grid_size(self):
        # expand set of inner, outer, divider cells
        for name in ["inner_cells", "outer_cells", "divider_cells"]:
            new_cells = set()
            for c in getattr(self, name):
                for x in range(2):
                    for y in range(2):
                        new_cells.add((c[0] * 2 + x, c[1] * 2 + y))
            setattr(self, name, new_cells)

        # reposition obstacles
        old_obstacles = self.obstacles
        self.obstacles = []

        # expand obstacles
        for o in old_obstacles:
            for x in range(2):
                for y in range(2):
                    self.obstacles.append(self.obstacle_cls())
                    self.obstacles[-1].init_pos = (
                        o.init_pos[0] * 2 + x,
                        o.init_pos[1] * 2 + y,
                    )

        # reposition objects
        for o in self.init_objects:
            o.init_pos = (o.init_pos[0] * 2, o.init_pos[1] * 2)

        # expand passages and openings
        # old_objects = self.init_objects
        # self.init_objects = []
        new_passages = []
        for o in self.init_objects:

            if not (isinstance(o, Passage) or isinstance(o, self.opening_cls)):
                continue

            expand_cells = set(
                [
                    (o.init_pos[0] + 0, o.init_pos[1] + 0),
                    (o.init_pos[0] + 1, o.init_pos[1] + 0),
                    (o.init_pos[0] + 0, o.init_pos[1] + 1),
                    (o.init_pos[0] + 1, o.init_pos[1] + 1),
                ]
            )

            for c in expand_cells:
                cc = (c[0] + o.dir_vec[0], c[1] + o.dir_vec[1])
                if cc in expand_cells:
                    o.init_pos = c
                    new_passages.append((Passage(o.dir_vec), cc))
                    expand_cells.remove(c)
                    expand_cells.remove(cc)
                    break

            for c in expand_cells:
                self.obstacles.append(self.obstacle_cls())
                self.obstacles[-1].init_pos = c

        for o, init_pos in new_passages:
            self.init_objects.add("passage", o, init_pos=init_pos)

        # reposition agent
        self.init_agent_pos = (self.init_agent_pos[0] * 2, self.init_agent_pos[1] * 2)
        # double agent view size
        self.agent_view_size *= 2

        # adjust grid size
        self.width *= 2
        self.height *= 2

        # double wall thickness
        self.obstacle_thickness *= 2

        return None

    def flip_vertical(self):
        for name in ["inner_cells", "outer_cells"]:
            new_cells = set()
            for c in getattr(self, name):
                new_cells.add((self.width - 1 - c[0], c[1]))
            setattr(self, name, new_cells)

        for o in self.init_objects:
            o.init_pos = (self.width - 1 - o.init_pos[0], o.init_pos[1])

        for o in self.obstacles:
            o.init_pos = (self.width - 1 - o.init_pos[0], o.init_pos[1])

        self.init_agent_pos = (
            self.width - 1 - self.init_agent_pos[0],
            self.init_agent_pos[1],
        )
        if self.init_agent_dir in [0, 2]:
            self.init_agent_dir = 2 - self.init_agent_dir

        for o in self.init_objects:
            if hasattr(o, "dir_vec"):
                o.dir_vec = (-o.dir_vec[0], o.dir_vec[1])

        return None

    def change_target_color(self):
        while True:
            new_color = self.random.choice(self.allowed_object_colors)
            if new_color != self.target_color:
                self.target_color = new_color
                break
        for o in self.init_targets:
            o.color = self.target_color
        self._set_mission()
        # change colors of distractors of the same type as target objects
        for o in self.init_distractors:
            if isinstance(o, type(self.init_targets[0])) and hasattr(o, "color"):
                o.color = self.random.choice(
                    list(set(self.allowed_object_colors) - set([self.target_color]))
                )
        return new_color

    def hide_target_in_box(self):
        boxes = []
        for o in self.init_targets:
            box = Box(color=self.random.choice(self.allowed_object_colors))
            box.contains = o
            self.init_objects.add("box", box, init_pos=o.init_pos)
            boxes.append(box)
        return boxes

    def change_agent_view_size(self):
        new_view = None
        while True:
            new_size = self.random.randint(
                self.agent_view_size - 2, self.agent_view_size + 2
            )
            if new_size != self.agent_view_size:
                break
        assert new_size is not None
        self.agent_view_size = new_size
        return new_size

    def add_opening(self):
        raise NotImplementedError

    def toggle_opening(self):
        raise NotImplementedError

    def add_passage(self):
        removed_obstacles, dir = self._try_drill_a_hole()
        self._remove_obstacles(removed_obstacles)
        for o in removed_obstacles:
            self.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)
        return removed_obstacles

    def block_opening(self):
        # find an opening that is not yet blocked
        free_openings = []
        for o in self.init_openings:
            is_blocked = False
            for oo in self.init_objects:
                if oo.cur_pos in self.outer_cells and oo.cur_pos in get_adjacent_cells(
                    o
                ):
                    is_blocked = True
                    break
            if not is_blocked:
                free_openings.append(o)
        if not free_openings:
            return None
        o = self.random.choice(free_openings)
        for c in get_adjacent_cells(o.init_pos):
            if c in self.outer_cells:
                color = self.random.choice(
                    list(set(self.allowed_object_colors) - set([self.target_color]))
                )
                ball = Ball(color=color)
                self.init_objects.add("blocking_ball", ball, init_pos=c)
                break
        return o, ball

    def put_agent_inside_section(self):
        occupied_cells = [o.init_pos for o in self.init_objects] + [
            o.init_pos for o in self.obstacles
        ]
        free_cells = list(set(self.inner_cells) - set(occupied_cells))
        self.init_agent_pos = self.random.choice(free_cells)
        return self.init_agent_pos

    def hide_tool_in_box(self):
        if not self.init_tools:
            return None
        # find a tool that is not already in a box
        target_tool = None
        while True:
            tool = self.random.choice(self.init_tools)
            is_inside_box = False
            for o in self.init_objects:
                if isinstance(o, Box) and o.contains == tool:
                    is_inside_box = True
            if not is_inside_box:
                target_tool = tool
                break

        if target_tool is None:
            return None

        box = Box(color=self.random.choice(self.allowed_object_colors))
        box.contains = target_tool
        self.init_objects.add("box", box, init_pos=target_tool.init_pos)

        return target_tool, box

    def remove_tool(self):
        if not self.init_tools:
            return None

        removed_tool = self.random.choice(self.init_tools)
        # remove tool
        self.init_objects.remove(self.tool_name, removed_tool)
        # remove box that holds tool
        removed_box = None
        if "box" in self.init_objects:
            for i, removed_box in enumerate(self.init_objects["box"]):
                if removed_box.contains == removed_tool:
                    self.init_objects.remove("box", removed_box)
                    break
        return removed_tool, removed_box

    def _try_drill_a_hole(self):
        def _drill_dir(o):
            # check if can drill into room
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (i == 0 or j == 0) and not (i == 0 and j == 0):
                        pos = (o.init_pos[0] + i * t, o.init_pos[1] + j * t)
                        if pos in self.inner_cells:
                            return i, j
            return None

        obstacles_and_dirs = []
        t = self.obstacle_thickness
        for o in self.obstacles:
            adjacent_to_outer = False
            nc = get_adjacent_cells(o.init_pos)
            for c in nc:
                if c in self.outer_cells:
                    adjacent_to_outer = True
                    break
            if not adjacent_to_outer:
                continue
            dir = _drill_dir(o)
            if dir is not None:
                obstacles_and_dirs.append((o, dir))
        no, dir = self.random.choice(obstacles_and_dirs)

        min_x = min(no.init_pos[0], no.init_pos[0] + dir[0] * (t - 1))
        max_x = max(no.init_pos[0], no.init_pos[0] + dir[0] * (t - 1))
        min_y = min(no.init_pos[1], no.init_pos[1] + dir[1] * (t - 1))
        max_y = max(no.init_pos[1], no.init_pos[1] + dir[1] * (t - 1))

        removed_obstacles = []
        for o in self.obstacles:
            if min_x <= o.init_pos[0] <= max_x and min_y <= o.init_pos[1] <= max_y:
                removed_obstacles.append(o)

        return removed_obstacles, dir

    def _remove_obstacles(self, removed_obstacles: List[WorldObj]):
        new_obstacles = []
        for o in self.obstacles:
            should_add = True
            for oo in removed_obstacles:
                if oo.init_pos == o.init_pos:
                    should_add = False
            if should_add:
                new_obstacles.append(o)
        self.obstacles = new_obstacles


class RoomDoorKeyEditor(BaseEditor):

    def add_opening(self):
        # find cell to put door
        removed_obstacles, dir = self._try_drill_a_hole()
        pos = None
        for i, o in enumerate(removed_obstacles):
            cn = get_adjacent_cells(o.init_pos)
            for n in cn:
                if n in self.outer_cells and pos is None:
                    pos = o.init_pos
            if pos is not None:
                break
        assert pos is not None

        # doors can't be open and locked (1, 1)
        door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
        door = self.opening_cls(
            self.init_doors[0].color,
            dir,
            is_open=door_state[0],
            is_locked=door_state[1],
        )
        self.init_objects.add("door", door, init_pos=pos)

        # create an entryway
        self._remove_obstacles(removed_obstacles)
        for o in removed_obstacles:
            if o.init_pos == door.init_pos:
                continue
            self.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)

        return door

    def toggle_opening(self):
        door_states = [(0, 0), (0, 1), (1, 0)]
        d = self.random.choice(self.init_doors)
        s = (d.is_open, d.is_locked)
        i = door_states.index(s)
        ns = door_states[(i + 1) % len(door_states)]
        d.is_open, d.is_locked = ns
        return d


class TreasureIslandEditor(BaseEditor):

    def add_opening(self):
        # find cell to put bridge
        pos = None
        removed_obstacles, dir = self._try_drill_a_hole()
        for o in removed_obstacles:
            cn = get_adjacent_cells(o.init_pos)
            for n in cn:
                if n in self.outer_cells and pos is None:
                    pos = o.init_pos
            if pos is not None:
                break
        assert pos is not None
        bridge = Bridge(dir, is_intact=self.random.choice([0, 1]))
        self.init_objects.add("bridge", bridge, init_pos=pos)

        # create an entryway
        self._remove_obstacles(removed_obstacles)
        for o in removed_obstacles:
            if o.init_pos == bridge.init_pos:
                continue
            self.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)
        return bridge

    def toggle_opening(self):
        b = self.random.choice(self.init_bridges)
        b.is_intact = not b.is_intact
        return b

    def make_lava_safe(self):
        self.obstacle_cls = SafeLava
        old_obstacles = self.obstacles
        self.obstacles = []
        for o in old_obstacles:
            self.obstacles.append(self.obstacle_cls())
            self.obstacles[-1].init_pos = o.init_pos
        return None

    def add_fireproof_shoes(self):
        occupied_cells = [o.init_pos for o in self.init_objects] + [
            o.init_pos for o in self.obstacles
        ]
        free_cells = list(set(self.outer_cells) - set(occupied_cells))
        shoes = FireproofShoes()
        self.init_objects.add(
            "fireproof_shoes", shoes, init_pos=self.random.choice(free_cells)
        )
        return shoes


class Edits(CustomEnum):

    # applicable to all environments
    none = "none"
    double_grid_size = "double_grid_size"
    flip_vertical = "flip_vertical"
    change_target_color = "change_target_color"
    hide_target_in_box = "hide_target_in_box"
    change_agent_view_size = "change_agent_view_size"
    add_opening = "add_opening"
    toggle_opening = "toggle_opening"
    add_passage = "add_passage"
    block_opening = "block_opening"
    put_agent_inside_section = "put_agent_inside_section"
    hide_tool_in_box = "hide_tool_in_box"
    remove_tool = "remove_tool"

    # treasure_island only
    make_lava_safe = "make_lava_safe"
    add_fireproof_shoes = "add_fireproof_shoes"
