from package.agents import Human, AI
from package.enums import MessageType
from package.infrastructure.config_utils import ConfigDict
from package.infrastructure.basic_utils import debug
from package.message import Message
#from package.builder import make_agents

from minigrid.minigrid_env import MiniGridEnv


class DiscussionGame:
    def __init__(self, config: ConfigDict):
        self.human = Human(config.human)
        self.ai = AI(config.ai)
        self.roles = config.roles
        self.observer = self.human if self.roles.observer == "human" else self.ai
        self.executor = self.human if self.roles.executor == "human" else self.ai
        self.solver = self.human if self.roles.solver == "human" else self.ai
        self.gt_env = self.observer.world_model
        self.config = config


class Game:
    def __init__(self, human: Human, ai: AI, ground_truth_env: MiniGridEnv = None):
        self.reset(human, ai, ground_truth_env)  # if env is none, use speaker's

    def reset(self, human: Human, ai: AI, ground_truth_env: MiniGridEnv):
        self.human = human
        self.ai = ai
        self.agent_mapping = {"p": self.human, "a": self.ai}
        self.gt_env = ground_truth_env

    def run(self, mismatch: str, first_speaker: str = "p", n_turns: int = 1):
        if first_speaker == "p":
            speaker = self.human
            listener = self.ai
            debug("Speaker = human, listener = ai")
            if self.gt_env is None:
                self.gt_env = self.human.world_model
        else:
            speaker = self.ai
            listener = self.human
            debug("Speaker = ai, listener = human")
            if self.gt_env is None:
                self.gt_env = self.ai.world_model
        if mismatch == "b":
            message = Message(MessageType.BELIEF_START)
        elif mismatch == "i":
            message = Message(MessageType.INTENTION_START)
        else:
            message = Message(MessageType.REWARD_START)
        debug("STARTING MESSAGE")
        debug(message)
        for _ in range(n_turns):
            message = speaker.speak(message)
            debug("SPEAKER SPOKE")
            debug(message)
            message = listener.listen(message)
            debug("LISTENER RESPONDED")
            debug(message)
            # speaker, listener = listener, speaker

    def evaluate(self, executor_agent: str = "a", evaluator_agent: str = "p"):
        executor = self.agent_mapping[executor_agent]
        evaluator = self.agent_mapping[evaluator_agent]

        self.gt_env.reset()
        done = False
        i = 0
        while not done and i < len(executor.policy):
            _, _, done, _, _ = self.gt_env.step(executor.policy[i])
            i += 1

        # is_solved = evaluator.verify(plan)
        # return plan, is_solved
        return True
