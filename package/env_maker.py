from constants import *
from minigrid.manual_control import ManualControl
from utils import *
from envs.go_to_task import GotoTask
from envs.pick_up_task import PickupTask
from envs.put_next_task import PutTask
from envs.collect_task import CollectTask
from envs.enums import *
from typing import List
import time
import random


def make_envs(task: EnvType,
              teacher_level: Level,
              student_level: Level = None,
              student_variants: List[Variant] = None,
              seed: int = None):
    assert EnvType.has_value(task), "Env type is not valid"
    assert Level.has_value(teacher_level), "Teacher level is not valid"
    assert student_level or student_variants, "Must have at least one of `student_level` or `student_variants`"
    if student_level:
        assert Level.has_value(student_level), "Student level is not valid"
    if student_variants:
        for sv in student_variants:
            assert Variant.has_value(sv), f"Student variant \"{sv}\" is not valid"

    if task == EnvType.GOTO:
        constructor = GotoTask
    elif task == EnvType.PICKUP:
        constructor = PickupTask
    elif task == EnvType.PUT:
        constructor = PutTask
    elif task == EnvType.COLLECT:
        constructor = CollectTask
    
    seed = random.randint(0, 10000) if not seed else seed
    teacher_env = constructor(seed, teacher_level)
    disallowed = {
        Variant.VIEW_SIZE: teacher_env.agent_view_size,
        Variant.COLOR: teacher_env.target_obj.color,
        Variant.ROOM_SIZE: teacher_env.room_size,
        Variant.NUM_OBJECTS: len(teacher_env.objs) - 1,
        Variant.NUM_ROOMS: teacher_env.num_rooms if hasattr(teacher_env, "num_rooms") else None,
    }
    if student_level and student_variants:
        student_env = constructor(seed,
                                  student_level,
                                  target_obj = type(teacher_env.target_obj),
                                  variants = student_variants,
                                  disallowed = disallowed)
    elif student_level:
        student_env = constructor(seed,
                                  student_level,
                                  target_obj = type(teacher_env.target_obj))
    elif student_variants:
        student_env = constructor(seed,
                                  teacher_env.level,
                                  target_obj = type(teacher_env.target_obj),
                                  variants = student_variants,
                                  disallowed = disallowed)

    return teacher_env, student_env


if __name__ == "__main__":
    teacher_env, student_env = make_envs(task = EnvType.GOTO,
                                         teacher_level = Level.HIDDEN_KEY,
                                         student_level = Level.EMPTY,
                                         student_variants = None,
                                         seed = 222)
    mc = ManualControl(teacher_env)
    mc.start()

    # while True:
    #     teacher_env.reset()
    #     time.sleep(1)
    #     teacher_env.step(1)
    #     time.sleep(1)
    #     teacher_env.step(2)
    #     time.sleep(1)