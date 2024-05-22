# MindGrid

We present MindGrid, a research toolkit for studying practical alignment in human-AI collaboration tasks via assistive communication processes.

## Installation
Clone the repository.
```
git clone ...
```
Create a new conda environment and install dependencies.
```
conda create -n mindgrid python=3.10
cd mindgrid
pip install -r requirements.txt
```

## Quickstart Example
Quickly play around with all that MindGrid has to offer with this quickstart example.
```
# Make sure you have defined a config YAML file, say `config.yaml`. Check out `mindgrid/configs/base.yaml` for inspiration!

from mindgrid.builder import make_env
from mindgrid.ifnrastructure.config_utils import make_config

with open("/path/to/config.yaml", "r") as f:
    config_str = f.read()

game_config = make_config(config_str = config_str)

env = make_env(getattr(game_config, game_config.roles.executor).world_model)

env.render_mode = "human"

env.render()
```

## Built-in Functionalities
### Skill Overview
- MiniGrid primitives: `forward`, `left`, `right`, `pickup`, `drop`, `toggle`
- Moving skills: `move_DIR_X_steps`. Ex. `move_right_3_steps`, `move_backward_5_steps`
- `go_to_COLOR_OBJECT`. Valid for all objects.
- `pickup_COLOR_OBJECT`. Valid for `Ball`, `Key`, `Box`.
- `put_down_COLOR_OBJECT`. Valid for `Ball`, `Key`, `Box`.
- `open_COLOR_OBJECT`. Valid for `Box`, `Door`.
- `close_COLOR_OBJECT`. Valid for `Door`.
- `unlock_COLOR_OBJECT`. Valid for `Door`.

### Reward Function Overview
Basic reward functions:
- `reward_reach_OBJECT`. Best for `Goto` task.
- `reward_carry_OBJECT`. Best for `Pickup` task.
- `reward_adjacent_objects`. Best for `Put`, `Collect`, and `Cluster` tasks.
Advanced reward functions:
- `reward_far_away_from_region`
- `reward_close_to_region`
- `reward_COLOR1_OBJECT1_over_COLOR2_OBJECT2`

### Environment Edit Overview
- All environments
  - `change_room_size`
  - `change_room_orientation`
  - `change_target_color`
  - `hide_targets`
  - `hide_keys`
  - `remove_keys`
  - `change_field_of_vision`
  - `toggle_doors`
- `Room_Door_Key` level:
  - `add_opening_to_wall` (includes parameter for adding door instead of hole)
  - `block_door`
  - `put_agent_in_room`
- `Treasure_Island` level:
  - `add_bridge`
  - `make_lava_safe`
  - `add_fireproof_shoes`
  - `put_agent_on_island`

## Recreating Experiments
Our experiments—and other LLM-based agent operations in MindGrid—require the use of LLM APIs. Before running any LLM-based code, create a file called `access_tokens.py` inside `mindgrid` to store your API keys. The file should look something like this:
```
OPENAI_KEY = "yourkeyhere"
SCALE_KEY = "yourkeyhere"
HUGGINGFACE_KEY = "yourkeyhere"
```

To recreate, for example, the LLM prompting experiment in the paper for the environment mismatch speaker task, using few-shot prompting with LLaMA 3 70B Instruct, run the following:
```
python3 baselines/prompted_llm/belief_baseline.py -s -id -m 0 -f
```
`belief_baseline.py` contains code for the environment mismatch experiments. `intention_baseline.py` in the same directory contains code for the skillset mismatch experiments.

Flags:
- `-s`: run the speaker task experiment
- `-l`: run the listener task experiment
- `-id`: use the in-distribution test set
- `-ood`: use the out-of-distribution test set
- `-m`: the model index to use. We set `0` to LLaMA 3 70B Instruct, `1` to Mixtral 8x7B Instruct, and `2` to Gemma 7B Instruct
- `-f`: use few-shot prompting (without this flag, zero-shot prompting is used)

Model outputs will be logged inside the respective `few_shot/` and `zero_shot/` directories in `prompted_llm/`.

To evaluate model performance, run something like
```
python3 baselines/prompted_llm/evaluate.py -b -s
```
This will automatically impute the scores into `scores.csv` and output images as well inside `prompted_llm/`.

Flags:
- `-b`: evaluate environment mismatch
- `-i`: evaluate skillset mismatch
- `-s`: evaluate the speaker task experiment
- `-l`: evaluate the listener task experiment
- `-o`: get an overview of average scores per (mismatch, task, model). This requires a complete `scores.csv`
