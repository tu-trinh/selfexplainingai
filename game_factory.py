from mindgrid.discussion import Discussion
from mindgrid.infrastructure.config_utils import make_config
from mindgrid.infrastructure.basic_utils import debug, format_seconds
from mindgrid.infrastructure.env_constants import (
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

    config = make_config(file_path="mindgrid/configs/base.yaml")
    print(config)
    game = Discussion(config)

    game.ai.world_model.reset()
    game.ai.world_model.render()
    #print(game.ai.world_model.gen_navigation_map())

    input()
    pass


if __name__ == "__main__":

    random.seed(100)
    COLORS = split_attributes(COLOR_NAMES)
    # TODO: what is the min value for steps?
    STEPS = split_attributes(list(range(2, MAX_ROOM_SIZE - 1)))

    create_skillset_mismatch_games()
