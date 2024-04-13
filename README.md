# Benchmark Name

[short introduction of the benchmark]

## Installation
[figure this out for later]

## Quickstart Example
Since there are many different tasks of different difficulties that control what the agents can communicate with each other, this is just an example showing one type of task where the principal tries to help the attendant find a plan for the attendant's mismatched environment.
```
# Make sure you have defined a YAML file, say `config.yaml`

from builder import *

principal, attendant = builder.make_agents("path/to/config.yaml")
builder.set_advanced_reward_functions("path/to/config.yaml", principal, attendant)  # optional

image = principal.speak(mode = "image")
differences = attendant.listen(image = image)
adapted_solution = principal.listen(differences)
trajectory = attendant.execute_actions(adapted_solution)
is_solved = principal.verify(trajectory)
```

## Configuring Tasks
All configuration details should be in `package/configs/`. Make a separate `.yaml` file for every mismatch scenario, where a scenario is composed of a pair of agents and their corresponding mismatched environments. The YAML file must have three main fields: `principal`, `attendant`, and `env_specs`. @Khanh: see the `test1_difficulty1.yaml` file as an example.

### Configuring Agents
This applies to the `principal` and `attendant` fields in the YAML. There are three required subfields:
- `basic_reward_functions`: This should be a list of dictionaries, one for each reward function you want your agent's world model to have. The dictionary should have at most two keys: "name", the name of the higher-order reward function you want, and optionally "amt", the reward value you want to grant when reward conditions are met. This is by default set to 1. Make sure that the higher-order reward function you specify in "name" is defined inside `reward_functions.py`.
- `basic_reward_weights`: This should be a list of numbers (ints and floats are both fine), one for each reward function above. Don't worry about normalizing them; the agent will take care of that for you when it calculates the weighted average of the reward of a state.
- `skills`: This should be the complete list of skills you want your agent to be able to do. All you have to make sure of is that each skill you list is actually listed inside `skills.txt`, otherwise the agent builder won't pick it up. Don't worry about thinking if the world model you're building will actually support a particular skill; this logic is handled for you.

There are also optional subfields:
- `query_source`: This should be a string that says where you want to query your model from. Currently it supports the following APIs: OpenAI, Scale LLM Engine, and HuggingFace. By default, this is the OpenAI API.
- `model_source`: This should be a string that says which LLM family you want to use (such as "GPT" or "Mistral"). By default, this uses the GPT family, GPT-3.5-turbo.
- `advanced_reward_functions`: This should be defined similarly to the `basic_reward_functions` above: dictionary with the "name" of the reward function HOF as well as key/value pairs for the other HOF parameters. HOWEVER, they are a little more involved. While basic reward functions take care of which elements of the world model or agent are rewarded for you, advanced reward functions require you to define exactly what objects/regions/actions/etc. you want rewarded. Hence setting these functions are done in a different function call than the other functions, because you need to have seen your environment first before you know what to pass into these functions.
- `advanced_reward_weights`: This should be a list of weights for each advanced reward function you want your agent to have. Again, don't worry about normalizing them.

### Configuring Environments
This applies to the `env_specs` field in the YAML. There are three required subfields:
- `task`: This should be a member of the `Task` enum (see `enums.py`)
- `principal_level`: This should be a member of the `Level` enum that you want the principal's world model to be
- `seed`: This is a seed that will control the configuration of both the principal and attendant world models

There are two "optional" subfields for configuring the attendant's world model ("optional" in that you must define at least one). You should only define one or the other (though if you REALLY want to, you can have both).
- `attendant_level`: This should be a member of the `Level` enum that you want the attendant's world model to be
- `attendant_variants`: This should be a list containing members of the `Variant` enum that you want the attendant's world model to have
- `principal_render_mode` and `attendant_render_mode`: These should be one of the render modes defined [here](https://gymnasium.farama.org/api/env/#MiniGridEnv.render). By default they are `None`

When you call `make_agents` from `builder.py`, it will automatically create the world models and attach them to the agents.

### Adding Your Own!
[something about this package provides but a (strong) skeleton] Feel free to customize anything you want.
- Reward functions: If you want to write your own reward function, do so in `reward_functions.py`. Reward functions are encapsulated by higher-order functions (HOF). The inner function that they return defines the logic of the reward function, while the outer HOF is responsible for making sure this logic is applicable to the specific world model it is attached to. Follow this same structure for best results.
- Skills: If you want to write your own skills, do so in `skills.py`. Like the above, skills are encapsulated by HOFs that define a generic skill inside which then get "personalized" depending on what objects the agent's world model has. After you define a higher-order skill function, make sure you include all possible renditions of that function inside `skills.txt`.

## Built-in Functionalities
### Skill Overview
- MiniGrid primitives: `forward`, `left`, `right`, `pickup`, `drop`, `toggle`
- Moving skills: `move_X_steps_DIR`. Ex. `move_3_steps_right`, `move_5_steps_backward`
- `go_to_COLOR_OBJECT`. Valid for all objects
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


[bibtex when we have it]
