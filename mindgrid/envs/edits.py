from __future__ import annotations

import re
from typing import List
from abc import ABC
from copy import deepcopy as dc

from minigrid.core.world_object import Ball, Box, WorldObj, Lava
from mindgrid.envs.objects import Bridge, FireproofShoes, Passage, SafeLava
from mindgrid.infrastructure.basic_utils import (
    CustomEnum,
    get_adjacent_cells,
    DeterministicRandom,
)
from mindgrid.infrastructure.env_constants import COLOR_NAMES


class EditError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message


def parse_from_description(sent, env, exact=False):

    def clean(sent, template):

        # remove `and` between column and row
        pattern = r"(column \d+)\s+and\s+(row \d+)"
        sent = re.sub(pattern, r"\1 \2", sent)

        # remove color from sentence that does not require color
        if "color" not in template:
            pattern = r'\b(' + '|'.join(COLOR_NAMES) + r')\b'
            result = re.sub(pattern, '', sent)
            sent = re.sub(' +', ' ', result)

        # change `a` to `the`
        if template.startswith("the") and sent.startswith("a"):
            sent = "the" + sent[1:]

        return sent

    for edit in Edits:
        edit_cls = edit.value
        template = edit_cls.get_template(env)
        if not exact:
            clean_sent = clean(sent, template)

        # Extract placeholders from the template
        placeholders = re.findall(r'\{(.*?)\}', template)
        data = {}

        # Build a regex pattern based on the template to match the input string x
        pattern = template
        for placeholder in placeholders:
            if placeholder in ["col", "row"]:
                pattern = pattern.replace(f'{{{placeholder}}}', f'(?P<{placeholder}>\d)')
            elif placeholder == "state":
                regex = "closed|open|intact|damaged"
            elif placeholder == "tool":
                regex = "key|hammer"
            elif placeholder == "color":
                regex = "|".join(COLOR_NAMES)
            else:
                regex = "\w+"
            match_regex = f'(?P<{placeholder}>{regex})'
            pattern = pattern.replace(f'{{{placeholder}}}', match_regex)
        match = re.search(pattern, clean_sent)
        if match:
            args = match.groupdict()
            for k in args:
                if k in ["row", "col"]:
                    args[k] = int(args[k])
            return edit_cls(env, args=args)
    return None


def _snake_to_pascal(snake_str):
    # Split the string by underscores
    components = snake_str.split("_")
    # Capitalize the first letter of each component
    return "".join(x.title() for x in components)


class BaseEdit(ABC):
    def __init__(self, env: MindGridEnv, args: Dict[str, Any] = None):
        self.env = env
        self.args = args
        #self.state = env.get_state()
        self.random = DeterministicRandom(env.seed + 3209)

    def apply(self):
        return NotImplementedError

    def verbalize(self, env: MindGridEnv):
        return NotImplementedError

    def _drill_dir(self, env, obj):
        # check if can drill into room
        t = env.obstacle_thickness
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 or j == 0) and not (i == 0 and j == 0):
                    pos = (obj.init_pos[0] + i * t, obj.init_pos[1] + j * t)
                    if pos in env.inner_cells:
                        return i, j
        return None

    def _try_drill_a_hole(self, env: MindGridEnv):
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
            dir = self._drill_dir(env, o)
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

    @classmethod
    def get_template(cls, env: MindGridEnv):
        return cls.TEMPLATE


class DynamicEdit(BaseEdit):
    def __init__(self, env: MindGridEnv, args: Dict[str, Any] = None):
        self.env = env
        self.args = args
        inner_class = getattr(
            self.__class__, _snake_to_pascal(env.layout_name) + "Edit"
        )
        self.instance = inner_class(env, args=args)
        self.TEMPLATE = self.instance.TEMPLATE

    def apply(self):
        return self.instance.apply()

    def verbalize(self):
        return self.instance.verbalize()

    @classmethod
    def describe(cls, env: MindGridEnv):
        inner_class = getattr(cls, _snake_to_pascal(env.layout_name) + "Edit")
        return inner_class.describe(env)

    @classmethod
    def get_template(cls, env: MindGridEnv):
        inner_class = getattr(cls, _snake_to_pascal(env.layout_name) + "Edit")
        return inner_class.TEMPLATE


# NOTE: not working at the moment
class DoubleGridSizeEdit(BaseEdit):
    TEMPLATE = "the grid size has been doubled"

    def verbalize(self):
        return self.TEMPLATE

    def describe(env: MindGridEnv):
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
    TEMPLATE = "the grid has been flipped along the vertical axis"

    def verbalize(self):
        return self.TEMPLATE

    def describe(env: MindGridEnv):
        return "Flip the grid along the vertical axis to create a mirror reflection of the original."

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
    TEMPLATE = "the color of the target object has been changed to {color}"

    def verbalize(self):
        return self.TEMPLATE.format(color=self.new_color)

    def describe(env: MindGridEnv):
        return "Change the color of the target object. Set other objects that is of the same type and have the new target color to a different color."

    def apply(self):
        env = self.env

        old_color = env.target_color

        # set new color
        if self.args is None:
            while True:
                new_color = self.random.choice(env.allowed_object_colors)
                if new_color != env.target_color:
                    break
        else:
            new_color = self.args["color"]
        self.new_color = env.target_color = new_color

        for o in env.init_targets:
            o.color = env.target_color

        env._set_mission()
        # change color of distractors that are of the same type and was of the same color as target object
        for o in env.init_distractors:
            if isinstance(o, type(env.init_targets[0])) and o.color == new_color:
                o.color = old_color


class HideTargetInBoxEdit(BaseEdit):
    TEMPLATE = "the target object has been hidden inside a box"

    def verbalize(self):
        return self.TEMPLATE

    def describe(env: MindGridEnv):
        return "Hide a target object inside a box. If there are multiple target objects, one is randomly selected."

    def apply(self):
        env = self.env
        o = self.random.choice(env.init_targets)
        # NOTE: box has the same color as target
        box = Box(color=env.init_targets[0].color)
        box.contains = o
        env.init_objects.add("box", box, init_pos=o.init_pos)
        self.obj = dc(o)
        self.box = dc(box)


class ChangeAgentViewSizeEdit(BaseEdit):
    TEMPLATE = "the view size of the agent has been changed to {size}"

    def verbalize(self):
        return self.TEMPLATE.format(size=self.new_view_size)

    def describe(env: MindGridEnv):
        return "Change the view size of the agent. Let x be the old view size. The new view size is randomly chosen from x - 2 to x + 2."

    def apply(self):
        env = self.env

        if self.args is None:
            while True:
                new_size = self.random.randint(
                    env.agent_view_size - 2, env.agent_view_size + 2
                )
                if new_size != env.agent_view_size:
                    break
        else:
            new_view = self.args["size"]

        self.new_view_size = env.agent_view_size = new_size


class AddOpeningEdit(DynamicEdit):
    class RoomDoorKeyEdit(BaseEdit):
        TEMPLATE = "a new {state} door has been installed at column {col} row {row}"

        def verbalize(self):
            state = None
            if self.door.is_open:
                state = "open"
            elif self.door.is_locked:
                state = "locked"
            else:
                state = "closed"
            return self.TEMPLATE.format(
                state=state, col=self.door.init_pos[0], row=self.door.init_pos[1]
            )

        def describe(env: MindGridEnv):
            return "Add a door to the wall connecting the inner and outer room. The door can be open, closed, or locked."

        def apply(self):
            env = self.env

            # find cell to put door
            if self.args is None:
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
                state = self.random.choice(((0, 0), (0, 1), (1, 0)))
            else:
                pos = (self.args["col"], self.args["row"])
                state = self.args["state"]

                if state == "open":
                    state = (1, 0)
                elif state == "locked":
                    state = (0, 1)
                elif state == "closed":
                    state = (0, 0)
                else:
                    raise EditError(f"Undefined state {state}")

                removed_obstacles = []
                # NOTE: this doesn't work with double_grid_size
                for o in env.obstacles:
                    if o.init_pos == pos:
                        removed_obstacles.append(o)
                # can't find obstacle
                if not removed_obstacles:
                    raise EditError("Obstacle does not exist")
                dir = self._drill_dir(env, removed_obstacles[0])
                if dir is None:
                    raise EditError("Invalid position")

            door = env.opening_cls(
                env.init_doors[0].color,
                dir,
                is_open=state[0],
                is_locked=state[1],
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
        TEMPLATE = (
            "a new {state} bridge has been constructed at column {col} row {row}"
        )

        def verbalize(self):
            state = None
            if self.bridge.is_intact:
                state = "intact"
            else:
                state = "damaged"
            return self.TEMPLATE.format(
                state=state, col=self.bridge.init_pos[0], row=self.bridge.init_pos[1]
            )

        def describe(env: MindGridEnv):
            return "Add a bridge that connects the island to the mainland. The bridge can be either damaged or intact."

        def apply(self):
            env = self.env
            # find cell to put bridge
            if self.args is None:
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
                state = self.random.choice([0, 1])
            else:
                pos = (self.args["col"], self.args["row"])
                state = self.args["state"]

                if state == "intact":
                    state = 1
                elif state == "damaged":
                    state = 0
                else:
                    raise EditError(f"Undefined state {state}")

                removed_obstacles = []
                # NOTE: this doesn't work with double_grid_size
                for o in env.obstacles:
                    if o.init_pos == pos:
                        removed_obstacles.append(o)
                if not removed_obstacles:
                    raise EditError("Obstacle not found")
                dir = self._drill_dir(env, removed_obstacles[0])
                if dir is None:
                    raise EditError("Invalid position")

            bridge = Bridge(dir, is_intact=state)
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
        TEMPLATE = "the door at column {col} row {row} is no longer in the original state"

        def verbalize(self):
            return self.TEMPLATE.format(
                col=self.old_door.init_pos[0], row=self.old_door.init_pos[1]
            )

        def describe(env: MindGridEnv):
            return "Set an existing door to a new state (open, closed, or locked). If multiple doors are present, one will be selected randomly."

        def apply(self):
            env = self.env

            if self.args is None:
                door = self.random.choice(env.init_doors)
            else:
                door = None
                pos = (self.args["col"], self.args["row"])
                for d in env.init_doors:
                    if d.init_pos == pos:
                        door = d
                        break
                if door is None:
                    raise EditError("Door not found")

            states = [(0, 0), (0, 1), (1, 0)]
            curr_state = (door.is_open, door.is_locked)
            i = states.index(curr_state)
            next_state = states[(i + 1) % len(states)]

            self.old_door = dc(door)
            door.is_open, door.is_locked = next_state
            self.new_door = dc(door)

    class TreasureIslandEdit(BaseEdit):
        TEMPLATE = "the bridge at column {col} row {row} is no longer in the original state"

        def verbalize(self):
            return self.TEMPLATE.format(
                col=self.new_bridge.init_pos[0],
                row=self.new_bridge.init_pos[1],
            )

        def describe(env: MindGridEnv):
            return "Set an existing bridge to a new state (intact or damaged). If multiple bridges exist, one is selected randomly."

        def apply(self):
            env = self.env

            if self.args is None:
                bridge = self.random.choice(env.init_bridges)
            else:
                pos = (self.args["col"], self.args["row"])
                bridge = None
                for b in env.init_bridges:
                    if b.init_pos == pos:
                        bridge = b
                        break
                if bridge is None:
                    raise EditError("Bridge not found")

            self.old_bridge = dc(bridge)
            bridge.is_intact = not bridge.is_intact
            self.new_bridge = dc(bridge)


class AddPassageEdit(BaseEdit):
    TEMPLATE = "there is a walkable passage at column {col} row {row}"

    def verbalize(self):
        pos = [o.init_pos for o in self.removed_obstacles]
        p = list(sorted(pos, key=lambda x: (x[0], x[1])))[0]
        return self.TEMPLATE.format(col=p[0], row=p[1])

    def describe(env: MindGridEnv):
        return "Add a walkable passage connecting the inner room or the island with the outer section. The location of the passage is randomly chosen."

    def apply(self):
        env = self.env
        # NOTE: this doesn't work with double_grid_size
        if self.args is None:
            removed_obstacles, dir = self._try_drill_a_hole(env)
        else:
            removed_obstacles = []
            pos = (self.args["col"], self.args["row"])
            for o in env.obstacles:
                if o.init_pos == pos:
                    removed_obstacles.append(o)
            if not removed_obstacles:
                raise EditError("Obstacle not found")
            dir = self._drill_dir(env, removed_obstacles[0])
            if dir is None:
                raise EditError("Invalid position")

        self._remove_obstacles(env, removed_obstacles)
        for o in removed_obstacles:
            env.init_objects.add("passage", Passage(dir), init_pos=o.init_pos)
        self.removed_obstacles = dc(removed_obstacles)


class BlockOpeningEdit(BaseEdit):
    TEMPLATE = "a {color} ball at column {col} row {row} is blocking a path to the target object"

    def verbalize(self):
        return self.TEMPLATE.format(
            color=self.blocking_o.color,
            col=self.blocking_o.init_pos[0],
            row=self.blocking_o.init_pos[1],
        )

    def describe(env: MindGridEnv):
        return "Block a door or a bridge with a ball, making it impossible to access it from the outer section of the grid. If multiple doors or bridges are present, one will be randomly selected."

    def apply(self):
        env = self.env
        # find an opening that is not yet blocked
        if self.args is None:
            free_openings = []
            for o in env.init_openings:
                is_blocked = False
                for oo in env.init_objects:
                    if (
                        oo.init_pos in env.outer_cells
                        and oo.init_pos in get_adjacent_cells(o.init_pos)
                    ):
                        is_blocked = True
                        break
                if not is_blocked:
                    free_openings.append(o)
            if not free_openings:
                return
            o = self.random.choice(free_openings)
            for c in get_adjacent_cells(o.init_pos):
                if c in env.outer_cells:
                    color = self.random.choice(
                        list(set(env.allowed_object_colors) - set([env.target_color]))
                    )
                    ball = Ball(color=color)
                    pos = c
                    break
        else:
            color = self.args["color"]
            ball = Ball(color=color)
            pos = (self.args["col"], self.args["row"])

        env.init_objects.add("blocking_ball", ball, init_pos=pos)
        self.blocking_o = dc(ball)


class PutAgentInsideSectionEdit(BaseEdit):
    TEMPLATE = "the agent's starting location has been moved to column {col} row {row}"

    def verbalize(self):
        return self.TEMPLATE.format(col=self.pos[0], row=self.pos[1])

    def describe(env: MindGridEnv):
        return "Put the agent within the island or the inner room."

    def apply(self):
        env = self.env
        if self.args is None:
            occupied_cells = [o.init_pos for o in env.init_objects] + [
                o.init_pos for o in env.obstacles
            ]
            free_cells = list(set(env.inner_cells) - set(occupied_cells))
            pos = self.random.choice(free_cells)
        else:
            pos = (self.args["col"], self.args["row"])
            if env.grid.get(*pos) is not None:
                raise EditError("Cell is occupied")

        self.pos = env.init_agent_pos = pos


class HideToolInBoxEdit(BaseEdit):
    TEMPLATE = "the {color} {tool} was hidden inside a box"

    def verbalize(self):
        if self.tool is None:
            return None
        return self.TEMPLATE.format(color=self.tool.color, tool=self.tool.type)

    def describe(env: MindGridEnv):
        return "Hide a key or a hammer inside a box. If there are multiple keys or hammers, randomly choose one from those that are not already hidden inside boxes."

    def apply(self):
        env = self.env
        # no tool, do nothing
        if not env.init_tools:
            self.tool = None
            return

        # find a tool that is not already in a box
        if self.args is None:
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
                return
        else:
            target_tool = None
            for tool in env.init_tools:
                if tool.type == self.args["tool"] and tool.color == self.args["color"]:
                    target_tool = tool
                    break
            if target_tool is None:
                raise EditError("Tool not found")
            for o in env.init_objects:
                if isinstance(o, Box) and o.contains == target_tool:
                    # tool is already inside a box
                    return

        self.tool = target_tool
        box = Box(color=env.init_targets[0].color)
        box.contains = target_tool
        env.init_objects.add("box", box, init_pos=target_tool.init_pos)

        #self.box = dc(box)


class RemoveToolEdit(BaseEdit):
    TEMPLATE = "the {color} {tool} has disappeared"

    def verbalize(self):
        if self.tool is None:
            return None
        return self.TEMPLATE.format(color=self.tool.color, tool=self.tool.type)

    def describe(env: MindGridEnv):
        return "Remove a key or hammer from the grid. If there are multiple keys or hammers, one is selected at random. If the removed object was hidden inside a box, the box is also removed."

    def apply(self):
        env = self.env
        if not env.init_tools:
            self.tool = None
            return None

        if self.args is None:
            removed_tool = self.random.choice(env.init_tools)
        else:
            removed_tool = None
            for tool in env.init_tools:
                if tool.type == self.args["tool"] and tool.color == self.args["color"]:
                    removed_tool = tool
                    break
            if removed_tool is None:
                raise EditError("Tool not found")

        # remove tool
        env.init_objects.remove(env.tool_name, removed_tool)
        # remove box that holds tool
        removed_box = None
        if "box" in env.init_objects:
            for i, removed_box in enumerate(env.init_objects["box"]):
                if removed_box.contains == removed_tool:
                    env.init_objects.remove("box", removed_box)
                    break

        self.tool = dc(removed_tool)


class MakeLavaSafeEdit(BaseEdit):
    TEMPLATE = "the lava is safe to walk on"

    def verbalize(self):
        return self.TEMPLATE

    @classmethod
    def describe(env: MindGridEnv):
        return "Make the lava safe to walk on. The agent will not die from walking on this type of lava."

    def apply(self):
        env = self.env
        if env.obstacle_cls != Lava:
            raise EditError("Obstacle must be lava")
        env.obstacle_cls = SafeLava
        old_obstacles = env.obstacles
        env.obstacles = []
        for o in old_obstacles:
            env.obstacles.append(env.obstacle_cls())
            env.obstacles[-1].init_pos = o.init_pos
        return None


class AddFireproofShoesEdit(BaseEdit):
    TEMPLATE = "there is a pair of fire-proof shoes at column {col} row {row}"

    def verbalize(self):
        return self.TEMPLATE.format(
            col=self.shoes.init_pos[0], row=self.shoes.init_pos[1]
        )

    def describe(env: MindGridEnv):
        return "Add fire-proof shoes to the grid. When the agent is carrying the shoes, it will not die from walking on regular lava. The location of the shoes is chosen randomly."

    def apply(self):
        env = self.env

        if self.args is None:
            occupied_cells = [o.init_pos for o in env.init_objects] + [
                o.init_pos for o in env.obstacles
            ]
            free_cells = list(set(env.outer_cells) - set(occupied_cells))
            pos = self.random.choice(free_cells)
        else:
            pos = (self.args["col"], self.args["row"])
            if env.grid.get(*pos) is not None or pos == env.init_agent_pos:
                raise EditError("Cell is occupied")

        shoes = FireproofShoes()
        env.init_objects.add("fireproof_shoes", shoes, init_pos=pos)
        self.shoes = dc(shoes)


class Edits(CustomEnum):
    # applicable to all environments
    # double_grid_size = DoubleGridSizeEdit
    flip_vertical = FlipVerticalEdit
    change_target_color = ChangeTargetColorEdit
    hide_target_in_box = HideTargetInBoxEdit
    # change_agent_view_size = ChangeAgentViewSizeEdit
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
