from __future__ import annotations

import random
from collections import Counter

from mindgrid.builder import make_env
from mindgrid.skills import Skills


class Planner:

    def __init__(self, env_config):
        self.env = make_env(env_config)
        self.random = random.Random(env_config.seed)

    def __call__(self, env: MindGridEnv, skillset: List[Skills], must_have_skill: Skills = None):
        t = env.solve_with_optimal_skills()
        if t.is_null:
            return None
        return self._generate(t, skillset, must_have_skill=must_have_skill)

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
        self, t: Trajectory, skillset: List[Skills], must_have_skill: Skills = None
    ) -> List[Tuple[Skills, Dict]]:

        def parse(i, s):
            if f[i][s] is not None:
                return f[i][s]

            f[i][s] = False

            shuffle_skillset = self.random.sample(skillset, len(skillset))
            sorted_skillset = sorted(shuffle_skillset, key=lambda x: skill_count[x])

            #for ss in sorted(self.random.sample(skillset, len(skillset)), key=lambda x: priority[x]):

            # NOTE: this encourages matching diverse skills
            for ss in sorted_skillset:
                # NOTE: this encourages matching high-level skills
                for j in range(i):
                #for j in self.random.sample(range(i), i):
                    if (j, i, s) not in cached:
                        cached[(j, i, s)] = self._recognize(t.slice(j, i), s.value)
                    a = cached[(j, i, s)]
                    if a is not None:
                        #skill_count[s] += 1
                        ret = parse(j, ss)
                        #skill_count[s] -= 1
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

        #skill_count = Counter()

        # skillset = [Skills.open_door.value, Skills.unblock.value, Skills.get_object.value]
        #skillset = [Skills.move_object, Skills.get_object, Skills.open_door]
        # skillset.remove(Skills.unblock.value)
        # skillset.remove(Skills.get_object.value)

        #for s in sorted(self.random.sample(skillset, len(skillset)), key=lambda x: priority[x]):

        if must_have_skill:
            for i in range(n):
                for j in range(0, i):
                    a = self._recognize(t.slice(j, i), must_have_skill.value)
                    if a is not None:
                        p1 = self._generate(t.slice(0, j), skillset) if j > 0 else []
                        p2 = self._generate(t.slice(i, n - 1), skillset) if i < n - 1 else []
                        if p1 is not None and p2 is not None:
                            return p1 + [(must_have_skill, a)] + p2
        else:
            skill_count = Counter()
            for s in self.random.sample(skillset, len(skillset)):
                ret = parse(n - 1, s)
                if ret != False:
                    return ret
        return None
