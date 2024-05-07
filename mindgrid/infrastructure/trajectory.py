from __future__ import annotations

from typing import Tuple


class Trajectory:

    def __init__(self):
        self.states, self.actions = [], []

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_actions(self):
        return len(self.actions)

    def add(self, state: MindGridEnvState, action: ActType = None):
        self.states.append(state)
        if action is not None:
            self.actions.append(action)

    def get(self, i: int) -> Tuple[MindGridEnvState, ActType]:
        if i >= self.n_actions:
            return self.states[i]
        return self.states[i], self.actions[i]

    def slice(self, start: int, end: int):
        assert end >= start, "end must be >= start"
        new_t = Trajectory()
        for i in range(start, end):
            new_t.add(*self.get(i))
        new_t.add(self.get(end))
        self.check(new_t)
        return new_t

    def merge(self, t: Trajectory) -> Trajectory:
        assert (
            self.last_state == t.first_state
        ), "Can't merge incompatible trajectories!"
        new_t = Trajectory()
        for i in range(self.n_states - 1):
            new_t.add(*self.get(i))
        for i in range(t.n_states - 1):
            new_t.add(*t.get(i))
        new_t.add(t.last_state)
        self.check(new_t)
        return new_t

    def check(self, t: Self):
        assert t.n_actions == t.n_states - 1

    @property
    def first_action(self):
        return self.actions[0]

    @property
    def last_action(self):
        return self.actions[-1]

    @property
    def first_state(self):
        return self.states[0]

    @property
    def last_state(self):
        return self.states[-1]
