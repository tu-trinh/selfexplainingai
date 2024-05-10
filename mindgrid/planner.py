from __future__ import annotations

import random
from collections import Counter

from mindgrid.builder import make_env
from mindgrid.skills import Skills


class Planner:

    """
    NO_REEXECUTION_SKILLS = [
        Skills.PRIMITIVE.value,
        Skills.GOTO.value,
        Skills.ROTATE_TOWARDS_OBJECT.value,
        Skills.ROTATE_TOWARDS_DIRECTION.value,
        Skills.GOTO_ADJACENT_OBJECT.value,
        Skills.GOTO_ADJACENT_POSITION.value
    ]
    """

    def __init__(self, env_config):
        self.env = make_env(env_config)
        self.env.reset()
        self.random = random.Random(env_config.seed)

    def __call__(self, env: MindGridEnv, skillset: List[BaseSkill]):
        t = env.solve_with_optimal_skills()
        if t is None:
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
        #print("----", tt.n_actions, t.n_actions, tt == t)
        if tt is None or tt != t:
            return None

        return a

    def _generate(self, t: Trajectory, skillset: List[BaseSkill]) -> List[Tuple[BaseSkill, Dict]]:

        def parse(i, s):
            if f[i][s] is not None:
                return f[i][s]

            shuffled_skillset = self.random.sample(skillset, len(skillset))
            sorted_skillset = sorted(shuffled_skillset, key=lambda x: skill_count[x])

            for ss in sorted_skillset:
                for j in self.random.sample(range(i), i):
                    if (j, i, s) not in cached:
                        cached[(j, i, s)] = self._recognize(t.slice(j, i), s)
                        if s == Skills.GET_OBJECT.value and cached[(j, i, s)] is not None:
                            print(j, i, s, cached[(j, i, s)], t.n_states)
                    a = cached[(j, i, s)]
                    if a is not None:
                        skill_count[s] += 1
                        ret = parse(j, ss)
                        skill_count[s] -= 1
                        if ret != False:
                            f[i][s] = ret + [(s, a)]
                            return f[i][s]

            """
            for j in self.random.sample(range(i), i):
            #for j in range(27, 28):
            #for j in range(i):
                if (j, i, s) not in cached:
                    print(j, i, s)
                    cached[(j, i, s)] = self._recognize(t.slice(j, i), s)
                a = cached[(j, i, s)]
                print(j, i, s, a)
                if a is not None:
                    shuffled_skillset = self.random.sample(skillset, len(skillset))
                    sorted_skillset = sorted(shuffled_skillset, key=lambda x: skill_count[x])
                    print(sorted_skillset)
                    #for ss in self.random.sample(skillset, len(skillset)):
                    for ss in sorted_skillset:
                        skill_count[s] += 1
                        ret = parse(j, ss)
                        skill_count[s] -= 1
                        if ret != False:
                            f[i][s] = ret + [(s, a)]
                            return f[i][s]
            """
            return False

        cached = {}

        del skillset[skillset.index(Skills.GET_OBJECT.value)]
        del skillset[skillset.index(Skills.GOTO.value)]
        del skillset[skillset.index(Skills.GO_DIR_N_STEPS.value)]

        #skillset = [Skills.GET_OBJECT.value, Skills.UNBLOCK.value, Skills.FIX_BRIDGE.value, Skills.OPEN_BOX.value]

        n = t.n_states
        f = [None] * n
        for i in range(n):
            f[i] = {}
            for s in skillset:
                f[i][s] = [] if i == 0 else None

        skill_count = Counter()

        for s in self.random.sample(skillset, len(skillset)):
        #for s in [Skills.GET_OBJECT.value]:
            ret = parse(n - 1, s)
            if ret != False:
                return ret
        return None



    """
    def _generate(self, t: Trajectory, skillset: List[BaseSkill]) -> List[Tuple[BaseSkill, Dict]]:
        # dynamic programming
        n = t.n_states
        f = [None] * n
        for i in range(n):
            f[i] = {}
            for s in skillset:
                f[i][s] = False

        for s in skillset:
            f[0][s] = True

        for i in range(n):
            for s in skillset:
                for j in range(i):
                    if self._recognize(t.slice(j, i), s) is not None:
                        for ss in skillset:
                            if f[j][ss]:
                                f[i][s] = True

        # generate plan
        plan = []
        skill_count = Counter()
        for s in self.random.sample(skillset, len(skillset)):
            if f[i][s]:
                self._trace_back(i, s, f, plan, skill_count, t, skillset)
                return plan
        return None

    def _trace_back(self, i, s, f, plan, skill_count, t, skillset):
        if i == 0:
            return
        for j in self.random.sample(list(range(i)), i):
            skill_args = self._recognize(t.slice(j, i), s)
            if skill_args is not None:
                shuffle_skills = self.random.sample(skillset, len(skillset))
                skills_sorted_by_count = sorted(shuffle_skills, key=lambda x: skill_count[x])
                # prioritize skills with less counts
                for ss in skills_sorted_by_count:
                    if f[j][ss]:
                        skill_count[s] += 1
                        self._trace_back(j, ss, f, plan, skill_count, t, skillset)
                        plan.append((s, skill_args))
                        return
    """
