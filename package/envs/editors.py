from package.infrastructure.basic_utils import get_adjacent_cells
from package.envs.modifications import Bridge, FireproofShoes, SafeLava

from minigrid.core.world_object import WorldObj, Wall, Box, Door, Key, Ball, Goal

import numpy as np

from typing import List


class BaseEditor:

    def none(self):
        pass

    def double_grid_size(self):
        self.width *= 2
        self.height *= 2
        self.obstacle_thickness *= 2

        # expand set of inner and outer cells
        for name in ["inner_cells", "outer_cells"]:
            new_cells = set()
            for c in getattr(self, name):
                for x in range(2):
                    for y in range(2):
                        new_cells.add((c[0] * 2 + x, c[1] * 2 + y))
            setattr(self, name, new_cells)


        # reposition obstacles
        old_obstacles = self.obstacles
        self.obstacles = []
        for o in old_obstacles:

            if self.layout_name == "room_door_key":
                openings = self.doors
            elif self.layout_name == "treasure_island":
                openings = self.bridges

            has_opening = False
            for d in openings:
                if d.init_pos == o.init_pos:
                    has_opening = True
                    break

            for x in range(2):
                for y in range(2):
                    if has_opening and (x == 0 and y == 1):
                        continue
                    self.obstacles.append(self.obstacle_cls())
                    self.obstacles[-1].init_pos = (
                        o.init_pos[0] * 2 + x,
                        o.init_pos[1] * 2 + y,
                    )
        # reposition objects
        for o in self.objects:
            o.init_pos = (o.init_pos[0] * 2, o.init_pos[1] * 2)
        # reposition agent
        self.agent_pos = (self.agent_pos[0] * 2, self.agent_pos[1] * 2)
        # double agent view size
        self.agent_view_size *= 2

    def flip_vertical(self):
        for name in ["inner_cells", "outer_cells"]:
            new_cells = set()
            for c in getattr(self, name):
                new_cells.add((self.width - 1 - c[0], c[1]))
            setattr(self, name, new_cells)

        for o in self.objects + self.obstacles:
            o.init_pos = (self.width - 1 - o.init_pos[0], o.init_pos[1])
        self.agent_pos = (self.width - 1 - self.agent_pos[0], self.agent_pos[1])
        if self.agent_dir in [0, 2]:
            self.agent_dir = 2 - self.agent_dir

    def change_target_color(self):
        while True:
            new_color = self.random.choice(self.allowed_object_colors)
            if new_color != self.target_color:
                self.target_color = new_color
                break
        for o in self.target_objects:
            o.color = self.target_color
        self._set_mission()
        # change colors of distractors of the same type as target objects
        for o in self.distractor_objects:
            if isinstance(o, type(self.target_objects[0])) and hasattr(o, "color"):
                o.color = self.random.choice(
                    list(set(self.allowed_object_colors) - set([self.target_color]))
                )

    def hide_targets_in_boxes(self):
        for o in self.target_objects:
            box = Box(color=self.random.choice(self.allowed_object_colors))
            box.contains = o
            box.init_pos = o.init_pos
            self.objects.append(box)

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

    def _try_drill_a_hole(self):
        def _drill_dir(o):
            # can't drill at openings
            if self.layout_name == "room_door_key":
                openings = self.doors
            elif self.layout_name == "treasure_island":
                openings = self.bridges

            has_opening = False
            for d in openings:
                if d.init_pos == o.init_pos:
                    has_opening = True
                    break

            if has_opening:
                return None

            # check if can drill into room
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (i == 0 or j == 0) and not (i == 0 and j == 0):
                        pos = (o.init_pos[0] + i * t, o.init_pos[1] + j * t)
                        if pos in self.inner_cells:
                            return -i, -j
            return None

        obstacles_and_dirs = []
        t = self.obstacle_thickness
        for o in self.obstacles:
            dir = _drill_dir(o)
            if dir is not None:
                obstacles_and_dirs.append((o, dir))
        no, dir = self.random.choice(obstacles_and_dirs)

        print("--", no.init_pos, dir)

        min_x = min(no.init_pos[0], no.init_pos[0] + dir[0] * (t - 1))
        max_x = max(no.init_pos[0], no.init_pos[0] + dir[0] * (t - 1))
        min_y = min(no.init_pos[1], no.init_pos[1] + dir[1] * (t - 1))
        max_y = max(no.init_pos[1], no.init_pos[1] + dir[1] * (t - 1))

        removed_obstacles = []
        for o in self.obstacles:
            if min_x <= o.init_pos[0] <= max_x and min_y <= o.init_pos[1] <= max_y:
                removed_obstacles.append(o)

        return removed_obstacles

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

    def put_agent_in_room(self):
        occupied_cells = [o.init_pos for o in self.objects + self.obstacles]
        free_cells = list(set(self.inner_cells) - set(occupied_cells))
        self.agent_pos = self.random.choice(free_cells)

    def add_opening_to_wall(self):
        self._remove_obstacles(self._try_drill_a_hole())

    def add_door(self):
        # doors can't be open and locked (1, 1)
        door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
        door = Door(
            color=self.doors[0].color, is_open=door_state[0], is_locked=door_state[1]
        )
        # find cell to put door
        removed_obstacles = self._try_drill_a_hole()
        for i, o in enumerate(removed_obstacles):
            cn = get_adjacent_cells(o.init_pos)
            for n in cn:
                if n in self.outer_cells and door.init_pos is None:
                    door.init_pos = o.init_pos
        assert door.init_pos is not None
        self.doors.append(door)
        self.objects.append(door)

        # create an entryway
        self._remove_obstacles(removed_obstacles)

    def hide_key_in_box(self):
        if self.keys:
            key = self.keys[0]
            box = Box(color=self.random.choice(self.allowed_object_colors))
            box.contains = key
            box.init_pos = key.init_pos
            self.objects.append(box)

    def remove_key(self):
        key = self.keys[0]
        # remove box that holds key
        for i, o in enumerate(self.objects):
            if isinstance(o, Box) and o.contains == key:
                del iself.objects[i]
                break
        # remove key
        for i, o in enumerate(self.objects):
            if isinstance(o, Key) and o.color == self.doors[0].color:
                del self.objects[i]
                break

    def block_doors(self):
        for d in self.doors:
            nc = get_adjacent_cells(d.init_pos)
            for n in nc:
                if n in self.outer_cells:
                    color = self.random.choice(
                        list(set(self.allowed_object_colors) - set([self.target_color]))
                    )
                    ball = Ball(color=color)
                    ball.init_pos = n
                    self.distractor_objects.append(ball)
                    self.objects.append(ball)
                    break

    def toggle_doors(self):
        door_states = [(0, 0), (0, 1), (1, 0)]
        for d in self.doors:
            s = (d.is_open, d.is_locked)
            print(s)
            i = door_states.index(s)
            ns = door_states[(i + 1) % len(door_states)]
            d.is_open, d.is_locked = ns


class TreasureIslandEditor(BaseEditor):

    def put_agent_on_island(self):
        occupied_cells = [o.init_pos for o in self.objects + self.obstacles]
        free_cells = list(set(self.inner_cells) - set(occupied_cells))
        self.agent_pos = self.random.choice(free_cells)

    def add_bridge(self):
        bridge = Bridge()
        # find cell to put bridge
        removed_obstacles = self._try_drill_a_hole()
        for o in removed_obstacles:
            cn = get_adjacent_cells(o.init_pos)
            for n in cn:
                if n in self.outer_cells and bridge.init_pos is None:
                    bridge.init_pos = o.init_pos
        assert bridge.init_pos is not None
        self.bridges.append(bridge)
        self.objects.append(bridge)

        # create an entryway
        self._remove_obstacles(removed_obstacles)

    def make_lava_safe(self):
        self.obstacle_cls = SafeLava
        old_obstacles = self.obstacles
        self.obstacles = []
        for o in old_obstacles:
            self.obstacles.append(self.obstacle_cls())
            self.obstacles[-1].init_pos = o.init_pos

    def add_fireproof_shoes(self):
        occupied_cells = [o.init_pos for o in self.objects + self.obstacles]
        free_cells = list(set(self.outer_cells) - set(occupied_cells))
        shoes = FireproofShoes()
        shoes.init_pos = self.random.choice(free_cells)
        self.objects.append(shoes)

    def block_bridges(self):
        for b in self.bridges:
            nc = get_adjacent_cells(b.init_pos)
            for n in nc:
                if n in self.outer_cells:
                    color = self.random.choice(
                        list(set(self.allowed_object_colors) - set([self.target_color]))
                    )
                    ball = Ball(color=color)
                    ball.init_pos = n
                    self.distractor_objects.append(ball)
                    self.objects.append(ball)
                    break
