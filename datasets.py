from package.enums import Task, Level


"""
Belief Mismatch Dataset
"""
# Tasks: go to, pick up, etc.
# Levels: empty, death, etc
# Skills: every agent has . . .


"""
Intention Mismatch Dataset

Train   Test
------------
A1 A2  A3 A4 \
B1 B2  B3 B4 / same layout, diff skills
------------
C1 C2  D1 D2 \
C3 C4  D3 D4 / diff layout, same skills
------------
E1 E2  F5 F6 \
E3 E4  F7 F8 / diff layout, diff skills
"""
INTENTION_TRAIN = {}
INTENTION_TEST = {}
# make_envs(task: Task,
#               principal_level: Level,
#               attendant_level: Level = None,
#               attendant_variants: List[Variant] = None,
#               attendant_edits: List[str] = None,
#               seed: int = None,
#               principal_render_mode: str = None,
#               attendant_render_mode: str = None):
for task in Task:
    for level in Level:
        pass


"""
Reward Mismatch Dataset
"""