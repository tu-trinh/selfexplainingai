from package.builder import make_env
from package.infrastructure.config_utils import make_config
from package.skills import *

config = make_config(file_path="package/configs/base.yaml")

env = make_env(config.ai.world_model)

env.reset()

#PrimitiveAction(env.actions.left)(env)
#GoTo((5, 5))(env)
#GoToAdjacentObject(env.bridges[0])(env)
#GoToAdjacentPosition(env.target_objects[0].init_pos)(env)
#RotateTowardsObject(env.target_objects[0])(env)
#RotateTowardsPosition(env.target_objects[0].cur_pos)(env)

#GetObject(env.tools[0])(env)
#for o in env.objects:
#    if o.type == "fireproof_shoes":
#        GetObject(o)(env)
#GetObject(env.target_objects[0])(env)
#Unblock(env.bridges[0])(env)
#FixBridge(env.bridges[0])(env)
#for o in env.objects:
#    if o.type == "box":
#        OpenBox(o)(env)
#        break
#OpenDoor(env.doors[1])(env)
#GetObject(env.target_objects[0])(env)

env.solve_with_optimal_skills()


"""
while True:

    env.render()
    print(env.agent_dir)
    c = input("Action: ")
    if c == "a":
        a = env.actions.left
    elif c == "d":
        a = env.actions.right
    elif c == "w":
        a = env.actions.forward
    elif c == "t":
        a = env.actions.toggle
    elif c == "r":
        a = env.actions.pickup
    elif c == "e":
        a = env.actions.drop
    elif c == "q":
        break
    else:
        print("Action is not valid!")


    obs, reward, terminated, truncated, info = env.step(a)

    if terminated or truncated:
        break
"""
