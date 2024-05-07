from mindgrid.infrastructure.config_utils import ConfigDict
from mindgrid.builder import make_env

class Agent:
    def __init__(self, config: ConfigDict):
        self.world_model = make_env(config.world_model)
        self.skillset = config.skillset
        self.preference = config.preference


class Human(Agent):
    def __init__(self, config: ConfigDict):
        self.name = "Human"
        super().__init__(config)


class AI(Agent):
    def __init__(self, config: ConfigDict):
        self.name = "AI"
        super().__init__(config)
