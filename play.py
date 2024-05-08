from mindgrid.builder import make_env
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.skills import *

config = make_config(file_path="mindgrid/configs/base.yaml")

env = make_env(config.ai.world_model)

env.reset()

#t = PrimitiveAction(env.actions.left)(env)
#print(PrimitiveAction.recognize(t))

#t = GoTo((5, 5))(env)
#print(GoTo.recognize(t))

#t = GoToAdjacentObject(env.doors[0])(env)
#print(GoToAdjacentObject.recognize(t))

#t = GoToAdjacentObject(env.bridges[0])(env)
#print(GoToAdjacentObject.recognize(t))

#t = GoToAdjacentPosition(env.target_objects[0].init_pos)(env)
#print(GoToAdjacentPosition.recognize(t))

#t = RotateTowardsObject(env.target_objects[0])(env)
#print(RotateTowardsObject.recognize(t))

#t = RotateTowardsDirection(3)(env)
#print(RotateTowardsDirection.recognize(t))

#t = GetObject(env.tools[0])(env)
#print(GetObject.recognize(t))
#for o in env.objects:
#    if o.type == "fireproof_shoes":
#        t = GetObject(o)(env)
#        print(GetObject.recognize(t))
#GetObject(env.target_objects[0])(env)

#t = Unblock(env.doors[0])(env)
#print(Unblock.recognize(t))

#t = Unblock(env.bridges[0])(env)
#print(Unblock.recognize(t))


#t = FixBridge(env.bridges[0])(env)
#print(FixBridge.recognize(t))

for o in env.objects:
    if o.type == "box":
        t = OpenBox(o)(env)
        print(OpenBox.recognize(t))
        break
#t = OpenDoor(env.doors[0])(env)
#print(OpenDoor.recognize(t))
#GetObject(env.target_objects[0])(env)


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
