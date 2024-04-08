from __future__ import annotations

from package.infrastructure.env_constants import ACTION_TO_IDX
from package.infrastructure.basic_utils import debug

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Wall

import copy
from collections import deque
import random


with open("package/configs/skills.txt", "r") as f:
    allowable_names = [skill.strip() for skill in f.readlines()]


class TaskNode:
    def __init__(self, name: str = ""):
        # assert name == "" or name in allowable_names, "Invalid name for task tree"  # FIXME: should be env allowable skills not all allowable skills, such as go to lava edge case
        self.name = name
        self.children = []
    
    def add_child(self, subtask: TaskNode, index: int = -1):
        if index == -1:
            self.children.append(subtask)
        else:
            self.children.insert(index, subtask)
    
    def update_name(self, new_name: str):
        self.name = new_name

    def execute(self, env: MiniGridEnv) -> WorldObj:
        # make sure env is copied and resetted BEFORE calling this function!!
        if len(self.children) == 0:
            env.step(ACTION_TO_IDX[self.name])
        else:
            for child in self.children:
                child.execute(env)
        agent_pos = env.agent_pos
        dir_vec = env.dir_vec
        facing_obj = env.grid.get(agent_pos[0] + dir_vec[0], agent_pos[1] + dir_vec[1])
        # if end of trajectory and agent is facing nothing or a wall, check if it's on a goal or a bridge
        if facing_obj is None or type(facing_obj) == Wall:
            facing_obj = env.grid.get(agent_pos[0], agent_pos[1])
        return facing_obj
    
    def __str__(self):
        if len(self.children) > 0:
            return self.name + "(CHILDREN: " + ", ".join([c.name for c in self.children]) + ")"
        return self.name

    def list_all_skills(self) -> List[str]:
        skills = []
        def helper(tree):
            queue = deque()
            queue.append(tree)
            while queue:
                node = queue.popleft()
                skills.append(node.name)
                for child in node.children:
                    queue.append(child)
        helper(self)
        return set([skill for skill in skills if " " not in skill])

    def get_skill_subset(self, alpha: float) -> List[str]:
        def helper(tree):
            subset = []
            stack = deque()
            stack.append(tree)
            while stack:
                node = stack.pop()
                if len(node.children) == 0:
                    subset.append(node.name)
                else:
                    traverse_further = random.random() < alpha  # higher alpha = higher chance of low level skills
                    if traverse_further:
                        for child in node.children[::-1]:  # so that skills are visited in order
                            stack.append(child)
                    else:
                        subset.append(node.name)
            return [subtask for subtask in subset if " " not in subtask]  # make sure to not include main task in skill list
        subset = []
        alpha -= 0.1
        while not subset and alpha < 1.0:
            alpha += 0.1
            subset = helper(self)
        max_move_dists = {"right": 0, "backward": 0, "left": 0, "forward": 0}
        for skill in subset:
            if skill.startswith("move"):
                direction = skill.split("_")[1]
                dist = int(skill.split("_")[2])
                max_move_dists[direction] = max(max_move_dists[direction], dist)
        for direction in max_move_dists:
            if max_move_dists[direction] > 0:
                for d in range(1, max_move_dists[direction]):
                    subset.append(f"move_{direction}_{d}_steps")
        return list(set(subset))
    