# Benchmark Name

[short introduction of the benchmark]

## Installation
[figure this out for later]
@Khanh: just git pull

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
- `task`: This should be a member of the `EnvType` enum (see `enums.py`)
- `principal_level`: This should be a member of the `Level` enum that you want the principal's world model to be
- `seed`: This is a seed that will control the configuration of both the principal and attendant world models

There are two "optional" subfields for configuring the attendant's world model ("optional" in that you must define at least one). You should only define one or the other (though if you REALLY want to, you can have both).
- `attendant_level`: This should be a member of the `Level` enum that you want the attendant's world model to be
- `attendant_variants`: This should be a list containing members of the `Variant` enum that you want the attendant's world model to have
- `principal_render_mode` and `attendant_render_mode`: These should be one of the render modes defined [here](https://gymnasium.farama.org/api/env/#gymnasium.Env.render). By default they are `None`

When you call `make_agents` from `builder.py`, it will automatically create the world models and attach them to the agents.

### Adding Your Own!
[something about this package provides but a (strong) skeleton] Feel free to customize anything you want.
- Reward functions: If you want to write your own reward function, do so in `reward_functions.py`. Reward functions are encapsulated by higher-order functions (HOF). The inner function that they return defines the logic of the reward function, while the outer HOF is responsible for making sure this logic is applicable to the specific world model it is attached to. Follow this same structure for best results.
- Skills: If you want to write your own skills, do so in `skills.py`. Like the above, skills are encapsulated by HOFs that define a generic skill inside which then get "personalized" depending on what objects the agent's world model has. After you define a higher-order skill function, make sure you include all possible renditions of that function inside `skills.txt`.

## @Khanh: alina how the heck is this code organized?
Great question! Inside the `selfexplainingai` directory itself, the only necessary file is `run.py` which is where I setup all the experiments and stuff.

Inside `package` are most of the files I mentioned before. In addition:
- `configs` directory: meant to hold `skills.txt` and any custom YAML files
- `envs` directory: holds all the environment making logic.
    - Each of the five tasks have their own file (e.g. `go_to_task.py`). `GotoTask` and `PickupTask` extend from `SingleTargetEnv` as they both only have one target object to interact with. `PutNextTask`, `CollectTask`, and `ClusterTask` extend from `MultiTargetEnv` as the agent needs to interact with multiple objects to succeed. Both `Single`- and `Multi-` target envs extend from `PragmaticEnv` just because there's lots of logic that overlaps. Also there is still some repetitive code, apologies, but you probably won't have to touch it. I can also modify anything you want me to
    - `modifications.py` stores Minigrid-style objects we want to have, right now it just has the door that can't be opened
    - `register_envs.py` is for registering the environments with gymnasium but I don't think we will use it
- `agents.py`: defines `Agent`, `Principal`, and `Attendant`. Each has the ability to add skills, add rewards, speak, and listen. `Principal` also additionally has the option to verify if the attendant has succeeded in the task
- `builder.py`: functions for creating the agents and environments
- `constants.py`: LLM queries, constants, etc.
- `enums.py`: enums for everything, including Task, Level, and Variant
- `llm.py`: handles querying LLMs
- `trajectories.py`: holds custom classes for Transitions and Trajectories
- `utils.py`: util functions, also functions to get descriptions of objects and environments


[bibtex when we have it]