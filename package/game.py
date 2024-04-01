from package.agents import Principal, Attendant
from package.enums import MessageType
from package.infrastructure.basic_utils import debug
from package.message import Message

from minigrid.minigrid_env import MiniGridEnv


class Game:
    def __init__(self, principal: Principal, attendant: Attendant, ground_truth_env: MiniGridEnv = None):
        self.reset(principal, attendant, ground_truth_env)  # if env is none, use speaker's

    def reset(self, principal: Principal, attendant: Attendant, ground_truth_env: MiniGridEnv):
        self.principal = principal
        self.attendant = attendant
        self.agent_mapping = {"p": self.principal, "a": self.attendant}
        self.gt_env = ground_truth_env

    def run(self, mismatch: str, first_speaker: str = "p", n_turns: int = 1):
        if first_speaker == "p":
            speaker = self.principal
            listener = self.attendant
            debug("Speaker = principal, listener = attendant")
            if self.gt_env is None:
                self.gt_env = self.principal.world_model
        else:
            speaker = self.attendant
            listener = self.principal
            debug("Speaker = attendant, listener = principal")
            if self.gt_env is None:
                self.gt_env = self.attendant.world_model
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
