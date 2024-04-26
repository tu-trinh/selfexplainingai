from package.infrastructure.config_utils import ConfigDict
from package.builder import make_env
from package.message import Message

class Agent:
    def __init__(self, config: ConfigDict):
        self.world_model = make_env(config.world_model)
        self.skillset = config.skillset
        self.preference = config.preference

    def speak(self, message: Message):
        pass

    def listen(self, message: Message):
        pass


class Human(Agent):
    def __init__(self, config: ConfigDict):
        self.name = "Human"
        super().__init__(config)


class AI(Agent):
    def __init__(self, config: ConfigDict):
        self.name = "AI"
        super().__init__(config)
