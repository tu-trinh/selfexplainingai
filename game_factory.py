from package.enums import Task, Level
from package.discussion import Discussion
from package.infrastructure.config_utils import make_config
from package.infrastructure.basic_utils import debug, format_seconds
from package.infrastructure.env_constants import (
    COLOR_NAMES,
    MIN_ROOM_SIZE,
    MAX_ROOM_SIZE,
    SKILL_PHRASES,
)

import random
from pprint import pprint
from typing import Union, List, Dict, Tuple


def split_attributes(
    attr: Union[List, Dict]
) -> Union[Tuple[List, List, List], Dict[str, List]]:

    def split_attr(attr: List):
        n = len(attr)
        # shuffle
        split_idx = random.sample(range(n), n)
        # split
        ret = {}
        ret["val"] = [attr[i] for i in split_idx[: n // 3]]
        ret["test"] = [attr[i] for i in split_idx[n // 3 : 2 * n // 3]]
        ret["train"] = [attr[i] for i in split_idx[2 * n // 3 :]]
        return ret

    ret = {}
    if isinstance(attr, list):
        ret = split_attr(attr)
    else:
        train_ret, val_ret, test_ret = {}, {}, {}
        for key in attr.keys():
            ret_val = split_attr(attr[key])
            for split in ret:
                ret[split][key] = ret_val[split]
    return ret


def create_skillset_mismatch_games():

    config = make_config(file_path="package/configs/base.yaml")
    print(config)
    game = Discussion(config)

    game.ai.world_model.reset()
    print('asfd', game.human.world_model.target_objects[0].color)
    print('sd', game.human.world_model.width)
    game.ai.world_model.render()

    input()
    pass


if __name__ == "__main__":

    random.seed(100)
    COLORS = split_attributes(COLOR_NAMES)
    # TODO: what is the min value for steps?
    STEPS = split_attributes(list(range(2, MAX_ROOM_SIZE - 1)))

    task_to_pref_mapping = {
        Task.GOTO: "reward_reach_object_hof",
        Task.PICKUP: "reward_carry_object_hof",
        Task.PUT: "reward_adjacent_object_hof",
        Task.COLLECT: "reward_adjacent_object_hof",
        Task.CLUSTER: "reward_adjacent_object_hof",
    }

    create_skillset_mismatch_games()
