from package.agents import Human, AI
from package.infrastructure.config_utils import ConfigDict


class Discussion:
    def __init__(self, config: ConfigDict):
        self.human = Human(config.human)
        self.ai = AI(config.ai)
        self.roles = config.roles
        self.observer = self.human if self.roles.observer == "human" else self.ai
        self.executor = self.human if self.roles.executor == "human" else self.ai
        self.solver = self.human if self.roles.solver == "human" else self.ai
        self.gt_env = self.observer.world_model
        self.config = config
