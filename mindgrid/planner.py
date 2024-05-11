from __future__ import annotations

import random
from collections import Counter

from mindgrid.builder import make_env
from mindgrid.skills import Skills


class Planner:

    def __init__(self, env_config):
        self.env = make_env(env_config)
        self.env.render_mode = None
        self.env.reset()
        self.random = random.Random(env_config.seed)

    def __call__(self, env: MindGridEnv, skillset: List[Skills]):
        t = env.solve_with_optimal_skills()
        if t.is_null:
            return None
        return self._generate(t, skillset)

    def _recognize(self, t: Trajectory, s: BaseSkill) -> bool:

        a = s.recognize(t)
        if a is None:
            return None

        #print(s, a, t.first_state.carrying)

        # re-execute skill to see if generated trajectory matches input trajectory
        self.env.reset_from_state(t.first_state)
        tt = s(**a)(self.env)
        # print("----", tt.n_actions, t.n_actions, tt == t)

        return a if tt == t else None

    def _generate(
        self, t: Trajectory, skillset: List[Skills]
    ) -> List[Tuple[Skills, Dict]]:

        def parse(i, s):
            if f[i][s] is not None:
                return f[i][s]

            shuffled_skillset = self.random.sample(skillset, len(skillset))
            sorted_skillset = sorted(shuffled_skillset, key=lambda x: skill_count[x])

            f[i][s] = False
            for ss in sorted_skillset:
                for j in self.random.sample(range(i), i):
                    if (j, i, s) not in cached:
                        cached[(j, i, s)] = self._recognize(t.slice(j, i), s.value)
                        # if (
                        #    s == Skills.get_object.value
                        #    and cached[(j, i, s)] is not None
                        # ):
                        #    print(j, i, s, cached[(j, i, s)]["obj"].color)
                    a = cached[(j, i, s)]
                    if a is not None:
                        skill_count[s] += 1
                        ret = parse(j, ss)
                        skill_count[s] -= 1
                        if ret != False:
                            f[i][s] = ret + [(s, a)]
                            return f[i][s]
            return f[i][s]

        cached = {}

        n = t.n_states
        f = [None] * n
        for i in range(n):
            f[i] = {}
            for s in skillset:
                f[i][s] = [] if i == 0 else None

        skill_count = Counter()

        # skillset = [Skills.open_door.value, Skills.unblock.value, Skills.get_object.value]
        # skillset = [Skills.unblock, Skills.get_object, Skills.open_door]
        # skillset.remove(Skills.unblock.value)
        # skillset.remove(Skills.get_object.value)

        for s in self.random.sample(skillset, len(skillset)):
            ret = parse(n - 1, s)
            if ret != False:
                return ret
        return None
