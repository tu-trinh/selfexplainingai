from agents import *
from message import *


class Game:
    def __init__(self, principal: Principal, attendant: Attendant):
        self.reset(principal, attendant)

    def reset(self, principal: Principal, attendant: Attendant):
        self.principal = principal
        self.attendant = attendant

    def run(self, mismatch: str, first_speaker: str = "p", n_turns: int = 1):
        if first_speaker == "p":
            speaker = self.principal
            listener = self.attendant
        else:
            speaker = self.attendant
            listener = self.principal
        if mismatch == "b":
            message = Message(MessageType.BELIEF_START)
        elif mismatch == "i":
            message = Message(MessageType.INTENTION_START)
        else:
            message = Message(MessageType.REWARD_START)
        for _ in range(n_turns):
            message = speaker.speak(message)
            message = listener.listen(message)
            # speaker, listener = listener, speaker
    
    def evaluate(self, executor_agent: str = "p"):
        if executor_agent == "p":
            executor, evaluator = self.principal, self.attendant
        else:
            executor, evaluator = self.attendant, self.principal
        plan = executor.plan()
        is_solved = evaluator.verify(plan)
        return plan, is_solved
