from package.agents import Principal, Attendant
from package.enums import MessageType
from package.infrastructure.basic_utils import debug


class Game:
    def __init__(self, principal: Principal, attendant: Attendant):
        self.reset(principal, attendant)

    def reset(self, principal: Principal, attendant: Attendant):
        self.principal = principal
        self.attendant = attendant
        self.agent_mapping = {"p": self.principal, "a": self.attendant}

    def run(self, mismatch: str, first_speaker: str = "p", n_turns: int = 1):
        if first_speaker == "p":
            speaker = self.principal
            listener = self.attendant
            debug("Speaker = principal, listener = attendant")
        else:
            speaker = self.attendant
            listener = self.principal
            debug("Speaker = attendant, listener = principal")
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
    
    def evaluate(self, executor_agent: str = "p", evaluator_agent: str = "a"):
        executor = self.agent_mapping[executor_agent]
        evaluator = self.agent_mapping[evaluator_agent]
        plan = executor.plan()
        is_solved = evaluator.verify(plan)
        return plan, is_solved
