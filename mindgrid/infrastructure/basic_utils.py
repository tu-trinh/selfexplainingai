from minigrid.core.world_object import WorldObj

import numpy as np
from enum import Enum
from typing import Union, List, Tuple, Dict


class CustomEnum(Enum):

    @classmethod
    def has_value(cls, value):
        if isinstance(value, Enum):
            return value in cls
        return any(value == item.name for item in cls)

def format_seconds(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    x1, y1 = p1
    x2, y2 = p2
    return abs(x2 - x1) + abs(y2 - y1)

def get_adjacent_cells(cell: Tuple[int, int], ret_as_list: bool = False) -> Union[set, List]:
    x, y = cell
    adj_cells = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    if ret_as_list:
        return adj_cells
    return set(adj_cells)

def get_diagonally_adjacent_cells(cell: Tuple[int, int]) -> set:
    x, y = cell
    return set([(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)])

def to_enum(enum: Enum, value: Union[List, str]) -> List:
    if isinstance(value, str):
        return enum[value]
    return [enum[val] for val in value]

def debug(*strs):
    if not strs:
        print()
    else:
        print("[DEBUG]", " ".join([str(s) for s in strs]))

def xor(*args, none_check: bool = True) -> bool:
    if none_check:
        boolean_arr = [arg is not None for arg in args]
    else:
        boolean_arr = [arg for arg in args]
    return np.count_nonzero(boolean_arr) == 1

def make_clusters(arr: List, num_clusters: int) -> List:
    clusters = [[] for _ in range(num_clusters)]
    for i, elem in enumerate(arr):
        clusters[i % num_clusters].append(elem)
    return clusters

def flatten_list(nested_list: List) -> List:
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

def compare_world_obj(obj1: WorldObj, obj2: WorldObj):
    if obj1 is None and obj2 is None:
        return True
    elif xor(obj1, obj2):
        return False
    basics = [type(obj1) == type(obj2), obj1.color == obj2.color, obj1.init_pos == obj2.init_pos]
    return all(basics)
    # if all(basics):
    #     # some objects have additional attributes where they might differ
    #     if type(obj1) == Door:
    #         extras = [obj1.is_open == obj2.is_open, obj1.is_locked == obj2.is_locked]
    #         return all(extras)
    #     elif type(obj1) == Box:
    #         extras = [compare_world_obj(obj1.contains, obj2.contains)]
    #         return all(extras)
    #     else:
    #         return True
    # else:
    #     return False
