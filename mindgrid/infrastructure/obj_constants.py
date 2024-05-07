from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Lava, Wall

from mindgrid.envs.objects import Bridge, FireproofShoes, HeavyDoor

PLAYABLE_OBJS = [Goal, Box, Ball]
TANGIBLE_OBJS = [Ball, Box]
DISTRACTOR_OBJS = [Box, Ball, Key]

OBJ_NAME_MAPPING = {
    Goal: "goal",
    Ball: "ball",
    Box: "box",
    Key: "key",
    Door: "door",
    Wall: "wall",
    Lava: "lava",
    HeavyDoor: "heavy_door",
    Bridge: "bridge",
    FireproofShoes: "fireproof_shoes",
}
NAME_OBJ_MAPPING = {v: k for k, v in OBJ_NAME_MAPPING.items()}
OBJ_PLURAL_MAPPING = {
    Goal: "goals",
    Ball: "balls",
    Box: "boxes",
    Key: "keys",
    Door: "doors",
    Wall: "walls",
    Lava: "lava",
    HeavyDoor: "heavy_doors",
    Bridge: "bridges",
    FireproofShoes: "fireproof_shoes",
}
