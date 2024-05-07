from gymnasium.envs.registration import register

tasks = ["GoTo", "PickUp", "PutNext", "Collect"]
levels = ["Plain", "Death", "Dist", "OpenDoor", "BlockedDoor", "UnlockDoor", "HiddenKey", "MultRooms", "Boss"]
variants = ["vColor", "vRoomSize", "vNumObjects"]

"""
GoTo Tasks
"""
register(
    id = "NAME-GoTo-Plain-Plain-v0",
    entry_point = "path.to:EnvClass",
    kwargs = dict
)

register(
    id = "NAME-GoTo-Plain-Death-v0",
    entry_point = "path.to:EnvClass",
    kwargs = dict
)

register(
    id = "NAME-GoTo-Plain-Dist-v0",
    entry_point = "path.to:EnvClass",
    kwargs = dict
)

register(
    id = "NAME-GoTo-Plain--v0",
    entry_point = "path.to:EnvClass",
    kwargs = dict
)