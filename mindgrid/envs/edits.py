from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod
from copy import deepcopy as dc
import inflect
import random

from minigrid.core.world_object import Ball, Box, WorldObj

from mindgrid.envs.objects import Bridge, FireproofShoes, Passage, SafeLava
from mindgrid.infrastructure.basic_utils import CustomEnum, get_adjacent_cells, to_enum
from mindgrid.infrastructure.env_utils import describe_object, describe_position, describe_object_state


def _snake_to_pascal(snake_str):
    # Split the string by underscores
    components = snake_str.split("_")
    # Capitalize the first letter of each component
    return "".join(x.title() for x in components)


class BaseEdit(ABC):

    def __init__(self, env: MindGridEnv):
        self.env = env
        self.state = env.get_state()
        self.random = random.Random(env.seed + 3209)

    def apply(self):
        return NotImplementedError

    def verbalize(self, env: MindGridEnv):
        return NotImplementedError

    def _try_drill_a_hole(self, env: MindGridEnv):
        def _drill_dir(o):
            # check if can drill into room
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if (i == 0 or j == 0) and not (i == 0 and j == 0):
                        pos = (o.init_pos[0] + i * t, o.init_pos[1] + j * t)
                        if pos in env.inner_cells:
                            return i, j
            return None

        obstacles_and_dirs = []
        t = env.obstacle_thickness
        for o in env.obstacles:
            adjacent_to_outer = False
            nc = get_adjacent_cells(o.init_pos)
            for c in nc:
                if c in env.outer_cells:
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
        for o in env.obstacles:
            if min_x <= o.init_pos[0] <= max_x and min_y <= o.init_pos[1] <= max_y:
                removed_obstacles.append(o)

        return removed_obstacles, dir

    def _remove_obstacles(self, env: MindGridEnv, removed_obstacles: List[WorldObj]):
        new_obstacles = []
        for o in env.obstacles:
            should_add = True
            for oo in removed_obstacles:
                if oo.init_pos == o.init_pos:
                    should_add = False
            if should_add:
                new_obstacles.append(o)
        env.obstacles = new_obstacles

    def _describe_object_full(self, o, article=None):
        o.cur_pos = o.init_pos
        return describe_object(o, self.state.objects, relative=False, partial=False, article=article)

    def _describe_object(self, o, article=None):
        o.cur_pos = o.init_pos
        return describe_object(o, self.state.objects, relative=False, partial=True, article=article)

    def _describe_position(self, pos):
        return describe_position(pos, self.state.full_obs.shape, relative=False)


class DynamicEdit(BaseEdit):

    def __init__(self, env: MindGridEnv):
        self.env = env
        inner_class = getattr(
            self.__class__, _snake_to_pascal(env.layout_name) + "Edit"
        )
        self.instance = inner_class(env)

    def apply(self):
        return self.instance.apply()

    def verbalize(self):
        return self.instance.verbalize()


class DoubleGridSizeEdit(BaseEdit):

    def verbalize(self):
        return "The grid size has been doubled."

    def describe():
        return "Double the dimensions of the grid. Every cell in the original grid is stretched into four cells. An object or entity at position (x, y) in the original grid is now roughly at position (2x, 2y). The width of the wall or lava stream is doubled."

    def apply(self):
        env = self.env
        # expand set of inner, outer, divider cells
        for name in ["inner_cells", "outer_cells", "divider_cells"]:
            new_cells = set()
            for c in getattr(env, name):
                for x in range(2):
                    for y in range(2):
                        new_cells.add((c[0] * 2 + x, c[1] * 2 + y))
            setattr(env, name, new_cells)

        # reposition obstacles
        old_obstacles = env.obstacles
        env.obstacles = []

        # expand obstacles
        for o in old_obstacles:
            for x in range(2):
                for y in range(2):
                    env.obstacles.append(env.obstacle_cls())
                    env.obstacles[-1].init_pos = (
                        o.init_pos[0] * 2 + x,
                        o.init_pos[1] * 2 + y,
                    )

        # reposition objects
        for o in env.init_objects:
            o.init_pos = (o.init_pos[0] * 2, o.init_pos[1] * 2)

        # expand passages and openings
        new_passages = []
        for o in env.init_objects:

            if not (isinstance(o, Passage) or isinstance(o, env.opening_cls)):
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
                env.obstacles.append(env.obstacle_cls())
                env.obstacles[-1].init_pos = c

        for o, init_pos in new_passages:
            env.init_objects.add("passage", o, init_pos=init_pos)

        # reposition agent
        env.init_agent_pos = (env.init_agent_pos[0] * 2, env.init_agent_pos[1] * 2)
        # double agent view size
        env.agent_view_size *= 2

        # adjust grid size
        env.width *= 2
        env.height *= 2

        # double wall thickness
        env.obstacle_thickness *= 2

        return None


class FlipVerticalEdit(BaseEdit):

    def verbalize(self):
        return "The grid has been flipped vertically."

    def describe():
        return "Flip the grid vertically to create a mirror reflection of the original."

    def apply(self):
        env = self.env

        for name in ["inner_cells", "outer_cells"]:
            new_cells = set()
            for c in getattr(env, name):
                new_cells.add((env.width - 1 - c[0], c[1]))
            setattr(env, name, new_cells)

        for o in env.init_objects:
            o.init_pos = (env.width - 1 - o.init_pos[0], o.init_pos[1])

        for o in env.obstacles:
            o.init_pos = (env.width - 1 - o.init_pos[0], o.init_pos[1])

        env.init_agent_pos = (
            env.width - 1 - env.init_agent_pos[0],
            env.init_agent_pos[1],
        )
        if env.init_agent_dir in [0, 2]:
            env.init_agent_dir = 2 - env.init_agent_dir

        for o in env.init_objects:
            if hasattr(o, "dir_vec"):
                o.dir_vec = (-o.dir_vec[0], o.dir_vec[1])

        return None


class ChangeTargetColorEdit(BaseEdit):

    def verbalize(self):
        target_type = self.state.objects["target"][0].type

        n_targets = len(self.state.objects["target"])
        old_targets = f"{self.old_color} {inflect.engine().plural(target_type, n_targets)}"
        have_targets = "have" if n_targets > 1 else "has"
        d = f"The {old_targets} {have_targets} changed color to {self.new_color}."

        n_distractors = self.n_changed_distractors
        if n_distractors > 0:
            old_distractors = f"{self.new_color} {inflect.engine().plural(target_type, n_distractors)}"
            have_distractors = "have" if n_distractors > 1 else "has"
            d += f" The {old_distractors} {have_distractors} changed color to {self.distractor_new_color}."
        return d


    def describe():
        return "Change the color of the target object. Set other objects that is of the same type and have the new target color to a different color."

    def apply(self):
        env = self.env

        self.old_color = env.target_color
        while True:
            new_color = self.random.choice(env.allowed_object_colors)
            if new_color != env.target_color:
                env.target_color = new_color
                break
        self.new_color = env.target_color

        for o in env.init_targets:
            o.color = env.target_color

        env._set_mission()
        # change colors of distractors of the same type as target objects
        distractor_new_color = self.random.choice(
            list(set(env.allowed_object_colors) - set([env.target_color]))
        )

        self.n_changed_distractors = 0
        for o in env.init_distractors:
            if isinstance(o, type(env.init_targets[0])) and hasattr(o, "color"):
                o.color = distractor_new_color
                self.n_changed_distractors += 1

        self.distractor_new_color = distractor_new_color


class HideTargetInBoxEdit(BaseEdit):

    def verbalize(self):
        obj = self._describe_object(self.obj)
        box = self._describe_object_full(self.box, article="a")
        return f"The {obj} is hidden inside {box}."

    def describe(self):
        return "Hide a target object inside a box. If there are multiple target objects, one is randomly selected."

    def apply(self):
        env = self.env
        o = self.random.choice(env.init_targets)
        box = Box(color=self.random.choice(env.allowed_object_colors))
        box.contains = o
        env.init_objects.add("box", box, init_pos=o.init_pos)
        self.obj = dc(o)
        self.box = dc(box)


class ChangeAgentViewSizeEdit(BaseEdit):

    def verbalize(self):
        return f"The view size of the agent is changed to {self.new_view_size}."

    def describe():
        return "Change the view size of the agent. Let x be the old view size. The new view size is randomly chosen from x - 2 to x + 2."

    def apply(self):
        env = self.env
        new_view = None
        while True:
            new_size = self.random.randint(
                env.agent_view_size - 2, env.agent_view_size + 2
            )
            if new_size != env.agent_view_size:
                break
        assert new_size is not None
        env.agent_view_size = new_size
        self.new_view_size = new_size
        return new_size


class AddOpeningEdit(DynamicEdit):

    class RoomDoorKeyEdit(BaseEdit):

        def verbalize(self):
            door = self._describe_object_full(self.door, article="a")
            return f"There is {door}."

        def describe():
            return "Add a door to the wall connecting the inner and outer room. The door can be open, closed, or locked."

        def apply(self):
            env = self.env
            # find cell to put door
            removed_obstacles, dir = self._try_drill_a_hole(env)
            pos = None
            for i, o in enumerate(removed_obstacles):
                cn = get_adjacent_cells(o.init_pos)
                for n in cn:
                    if n in env.outer_cells and pos is None:
                        pos = o.init_pos
                if pos is not None:
                    break
            assert pos is not None

            # doors can't be open and locked (1, 1)
            door_state = self.random.choice(((0, 0), (0, 1), (1, 0)))
            door = env.opening_cls(
                env.init_doors[0].color,
                dir,
                is_open=door_state[0],
                is_locked=door_state[1],
            )
            env.init_objects.add("door", door, init_pos=pos)

            # create an entryway
            self._remove_obstacles(env, removed_obstacles)
            for o in removed_obstacles:
                if o.init_pos == door.init_pos:
                    continue
                env.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)

            self.door = dc(door)

    class TreasureIslandEdit(BaseEdit):

        def verbalize(self):
            bridge = self._describe_object_full(self.bridge, article="a")
            return f"There is {bridge}."

        def describe():
            return "Add a bridge that connects the island to the mainland. The bridge can be either damaged or intact."

        def apply(self):
            env = self.env
            # find cell to put bridge
            pos = None
            removed_obstacles, dir = self._try_drill_a_hole(env)
            for o in removed_obstacles:
                cn = get_adjacent_cells(o.init_pos)
                for n in cn:
                    if n in env.outer_cells and pos is None:
                        pos = o.init_pos
                if pos is not None:
                    break
            assert pos is not None
            bridge = Bridge(dir, is_intact=self.random.choice([0, 1]))
            env.init_objects.add("bridge", bridge, init_pos=pos)

            # create an entryway
            self._remove_obstacles(env, removed_obstacles)
            for o in removed_obstacles:
                if o.init_pos == bridge.init_pos:
                    continue
                env.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)

            self.bridge = dc(bridge)


class ToggleOpeningEdit(DynamicEdit):

    class RoomDoorKeyEdit(BaseEdit):

        def verbalize(self):
            old_door = self._describe_object(self.old_door)
            new_door_state = describe_object_state(self.new_door)
            return f"The {old_door} is now {new_door_state}."

        def describe():
            return "Set an existing door to a new state (open, closed, or locked). If multiple doors are present, one will be selected randomly."

        def apply(self):
            env = self.env
            door_states = [(0, 0), (0, 1), (1, 0)]
            d = self.random.choice(env.init_doors)
            s = (d.is_open, d.is_locked)
            i = door_states.index(s)
            ns = door_states[(i + 1) % len(door_states)]

            self.old_door = dc(d)
            d.is_open, d.is_locked = ns
            self.new_door = dc(d)

            return d

    class TreasureIslandEdit(BaseEdit):

        def verbalize(self):
            old_bridge = self._describe_object(self.old_bridge)
            new_bridge_state = describe_object_state(self.new_bridge)
            return f"The {old_bridge} is now {new_bridge_state}."

        def describe():
            return "Set an existing bridge to a new state (intact or damaged). If multiple bridges exist, one is selected randomly."

        def apply(self):
            env = self.env
            b = self.random.choice(env.init_bridges)
            self.old_bridge = dc(b)
            b.is_intact = not b.is_intact
            self.new_bridge = dc(b)
            return b


class AddPassageEdit(BaseEdit):

    def verbalize(self):
        pos = [o.init_pos for o in self.removed_obstacles]
        pos_x = sorted([p[0] for p in pos])
        pos_y = sorted([p[1] for p in pos])
        if pos_x[0] == pos_x[-1]:
            at = ("column", pos_x[0])
            from_to = ("row", pos_y[0], pos_y[-1])
        else:
            at = ("row", pos_y[0])
            from_to = ("column", pos_x[0], pos_x[-1])
        return f"There is a walkable passage at {at[0]} {at[1]} from {from_to[0]} {from_to[1]} to {from_to[0]} {from_to[2]}"

    def describe():
        return "Add a walkable passage connecting the inner room or the island with the outer section. The location of the passage is randomly chosen."

    def apply(self):
        env = self.env
        removed_obstacles, dir = self._try_drill_a_hole(env)
        self._remove_obstacles(env, removed_obstacles)
        for o in removed_obstacles:
            env.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)
        self.removed_obstacles = dc(removed_obstacles)


class BlockOpeningEdit(BaseEdit):

    def verbalize(self):
        blocked_o = self._describe_object(self.blocked_o)
        blocking_o = self._describe_object_full(self.blocking_o, article="a")
        return f"The {blocked_o} is blocked by {blocking_o}."

    def describe():
        return "Block a door or a bridge with a ball, making it impossible to access it from the outer section of the grid. If multiple doors or bridges are present, one will be randomly selected."

    def apply(self):
        env = self.env
        # find an opening that is not yet blocked
        free_openings = []
        for o in env.init_openings:
            is_blocked = False
            for oo in env.init_objects:
                if oo.cur_pos in env.outer_cells and oo.cur_pos in get_adjacent_cells(
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
            if c in env.outer_cells:
                color = self.random.choice(
                    list(set(env.allowed_object_colors) - set([env.target_color]))
                )
                ball = Ball(color=color)
                env.init_objects.add("blocking_ball", ball, init_pos=c)
                break
        self.blocked_o = dc(o)
        self.blocking_o = dc(ball)


class PutAgentInsideSectionEdit(BaseEdit):

    def verbalize(self):
        pos = self._describe_position(self.pos)
        if self.env.layout_name == "room_door_key":
            return f"The agent starts from within the island at {pos}."
        else:
            return f"The agent starts from within the inner room at {pos}."

    def describe():
        return "Put the agent within the island or the inner room."

    def apply(self):
        env = self.env
        occupied_cells = [o.init_pos for o in env.init_objects] + [
            o.init_pos for o in env.obstacles
        ]
        free_cells = list(set(env.inner_cells) - set(occupied_cells))
        env.init_agent_pos = self.random.choice(free_cells)
        self.pos = env.init_agent_pos


class HideToolInBoxEdit(BaseEdit):

    def verbalize(self):
        if self.box is None:
            return f"There are no more {inflect.engine().plural(self.env.tool_name)} to hide on this grid."
        box = self._describe_object_full(self.box, article="a")
        return f"The {self.env.tool_name} is hidden inside {box}."

    def describe():
        return "Hide a key or a hammer inside a box. If there are multiple keys or hammers, randomly choose one from those that are not already hidden inside boxes."

    def apply(self):
        env = self.env
        if not env.init_tools:
            self.box = None
            return None
        # find a tool that is not already in a box
        target_tool = None
        while True:
            tool = self.random.choice(env.init_tools)
            is_inside_box = False
            for o in env.init_objects:
                if isinstance(o, Box) and o.contains == tool:
                    is_inside_box = True
            if not is_inside_box:
                target_tool = tool
                break

        if target_tool is None:
            return None

        box = Box(color=self.random.choice(env.allowed_object_colors))
        box.contains = target_tool
        env.init_objects.add("box", box, init_pos=target_tool.init_pos)

        self.box = dc(box)


class RemoveToolEdit(BaseEdit):

    def verbalize(self):
        removed_tool = self._describe_object_full(self.removed_tool)
        return f"The {removed_tool} is removed from the grid."

    def describe():
        return "Remove a key or hammer from the grid. If there are multiple keys or hammers, one is selected at random. If the removed object was hidden inside a box, the box is also removed."

    def apply(self):
        env = self.env
        if not env.init_tools:
            return None

        removed_tool = self.random.choice(env.init_tools)
        # remove tool
        env.init_objects.remove(env.tool_name, removed_tool)
        # remove box that holds tool
        removed_box = None
        if "box" in env.init_objects:
            for i, removed_box in enumerate(env.init_objects["box"]):
                if removed_box.contains == removed_tool:
                    env.init_objects.remove("box", removed_box)
                    break

        self.removed_tool = dc(removed_tool)


class MakeLavaSafeEdit(BaseEdit):

    def verbalize(self):
        return f"The lava is safe to walk on."

    def describe():
        return "Make the lava safe to walk on. The agent will not die from walking on this type of lava."

    def apply(self):
        env = self.env
        env.obstacle_cls = SafeLava
        old_obstacles = env.obstacles
        env.obstacles = []
        for o in old_obstacles:
            env.obstacles.append(env.obstacle_cls())
            env.obstacles[-1].init_pos = o.init_pos
        return None


class AddFireproofShoesEdit(BaseEdit):

    def verbalize(self):
        shoes = self._describe_object_full(self.shoes, article="a")
        return f"There is {shoes}."

    def describe():
        return "Add fire-proof shoes to the grid. When the agent is carrying the shoes, it will not die from walking on regular lava. The location of the shoes is chosen randomly."

    def apply(self):
        env = self.env
        occupied_cells = [o.init_pos for o in env.init_objects] + [
            o.init_pos for o in env.obstacles
        ]
        free_cells = list(set(env.outer_cells) - set(occupied_cells))
        shoes = FireproofShoes()
        env.init_objects.add(
            "fireproof_shoes", shoes, init_pos=self.random.choice(free_cells)
        )
        self.shoes = dc(shoes)


class Edits(CustomEnum):

    # applicable to all environments
    double_grid_size = DoubleGridSizeEdit
    flip_vertical = FlipVerticalEdit
    change_target_color = ChangeTargetColorEdit
    hide_target_in_box = HideTargetInBoxEdit
    change_agent_view_size = ChangeAgentViewSizeEdit
    add_opening = AddOpeningEdit
    toggle_opening = ToggleOpeningEdit
    add_passage = AddPassageEdit
    block_opening = BlockOpeningEdit
    put_agent_inside_section = PutAgentInsideSectionEdit
    hide_tool_in_box = HideToolInBoxEdit
    remove_tool = RemoveToolEdit

    # treasure_island only
    make_lava_safe = MakeLavaSafeEdit
    add_fireproof_shoes = AddFireproofShoesEdit
