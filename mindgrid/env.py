from __future__ import annotations

import random
from typing import List
from copy import deepcopy as dc
from types import SimpleNamespace

import numpy as np
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv

from mindgrid.envs.editors import Edit
from mindgrid.envs.layouts import Layout
from mindgrid.envs.objects import FireproofShoes
from mindgrid.envs.tasks import Task
from mindgrid.infrastructure.env_constants import COLOR_NAMES


class MindGridEnv(MiniGridEnv):
    def __init__(
        self,
        seed: int,
        task: Task,
        layout: Layout,
        edits: List[Edit],
        allowed_object_colors: List[str] = COLOR_NAMES,
        max_steps: int = 1000,
        agent_view_size: int = 5,
        render_mode=None,
        **kwargs,
    ):

        self.seed = seed
        self.random = random.Random(seed)

        self.task = task
        self.layout = layout
        self.edits = edits
        self.allowed_object_colors = allowed_object_colors
        self.room_size = 11
        self.is_single_target = True

        # Generate task mission space and super init
        mission_space = task.value.mission_space
        super().__init__(
            mission_space=mission_space,
            grid_size=self.room_size,
            max_steps=max_steps,
            see_through_walls=False,
            render_mode=render_mode,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        # Unique environment configs
        self.env_id = f"{self.task}-{self.layout}-{self.seed}"

    def reset(self, seed=None, options=None):

        self._gen_grid(self.width, self.height)

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def _gen_grid(self, width, height):
        # create grid
        self.grid = Grid(width, height)
        for i in range(self.obstacle_thickness):
            self.grid.wall_rect(i, i, width - i * 2, height - i * 2)
        # set obstacles
        for o in self.obstacles:
            self.grid.set(o.init_pos[0], o.init_pos[1], o)
        # set objects
        for o in self.objects:
            self.put_obj(o, o.init_pos[0], o.init_pos[1])
        # set agent
        self.agent_dir = self.agent_init_dir
        self.agent_pos = self.agent_init_pos

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            # NOTE: terminte when stepping into lava only when without fireproof shoes
            if fwd_cell is not None and (
                fwd_cell.type == "lava"
                and not isinstance(self.carrying, FireproofShoes)
            ):
                terminated = True
            # NOTE: terminate when bridge is broken
            if (
                fwd_cell is not None
                and fwd_cell.type == "bridge"
                and not fwd_cell.is_intact
            ):
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = (-1, -1)
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = tuple(fwd_pos)
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def gen_simple_2d_map(self):
        ret = np.zeros((self.width, self.height), dtype=np.int32)
        for i in range(self.width):
            for j in range(self.height):
                o = self.grid.get(i, j)
                if o is not None:
                    ret[i][j] = not o.can_overlap()
                    if o.type == "bridge":
                        ret[i][j] = not o.is_intact
                    if o.type == "lava":
                        ret[i][j] = not isinstance(self.carrying, FireproofShoes)
        return ret

    def get_state(self):
        return MindGridEnvState(self)


class MindGridEnvState:

    def __init__(self, env: MindGridEnv):
        self.objects = dc(env.objects)
        self.agent_dir = dc(env.agent_dir)
        self.agent_pos = tuple(dc(env.agent_pos))
        self.front_pos = tuple(dc(env.front_pos))
        self.dir_vec = tuple(dc(env.dir_vec))
        self.carrying = dc(env.carrying)
        self.outer_cells = dc(env.outer_cells)
        # TODO: add more if needed

    def __eq__(self, other):
        if self.agent_dir != other.agent_dir:
            print("agent_dir", self.agent_dir, other.agent_dir)
            return False
        if self.agent_pos != other.agent_pos:
            print("agent_pos")
            return False
        if self.front_pos != other.front_pos:
            print("front_pos")
            return False
        if self.dir_vec != other.dir_vec:
            print("dir_vec")
            return False
        if not self._are_objects_equal(self.carrying, other.carrying):
            print("carrying")
            return False
        for i, o in enumerate(self.objects):
            oo = other.objects[i]
            if not self._are_objects_equal(o, oo):
                print("object")
                return False
        # NOTE: assume that outer_cells never change
        return True

    def _are_objects_equal(self, o, oo):
        if o is None and oo is None:
            return True
        if o is None or oo is None:
            return False
        if type(o) != type(oo):
            return False
        if o.color != oo.color:
            return False
        if not self._are_objects_equal(o.contains, oo.contains):
            return False
        if o.init_pos != oo.init_pos:
            return False
        if o.cur_pos != oo.cur_pos:
            return False
        return True
