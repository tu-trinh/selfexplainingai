from package.envs. modifications import HeavyDoor, Bridge, FireproofShoes

from minigrid.core.world_object import Door, Key, Goal, Wall, Lava, Ball, Box


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
    FireproofShoes: "fireproof_shoes"
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
    FireproofShoes: "fireproof_shoes"
}
