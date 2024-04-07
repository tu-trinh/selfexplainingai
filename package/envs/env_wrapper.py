from minigrid.minigrid_env import MiniGridEnv

from typing import Dict


class EnvironmentWrapper:
    def __init__(self, arguments: Dict, env: MiniGridEnv = None):
        self.seeds_and_envs = {}
        self.custom_env_type = type(env)
        self.arguments = arguments
        if env is not None:
            self.seeds_and_envs[env.env_seed] = env
    

    def reset(self, seed: int):
        for s, e in self.seeds_and_envs.items():
            if seed == s:
                return e.reset()
        sprout = self.custom_env_type(seed, **self.arguments)
        sprout.set_allowable_skills()
        sprout.bind_wrapper(self)
        self.seeds_and_envs[seed] = sprout
        return sprout.reset()
    

    def retrieve_seeded_env(self, seed: int):
        return self.seeds_and_envs[seed]
