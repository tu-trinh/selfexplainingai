import os
import sys


class Game:

    def reset(self, principal, attendant):
        self.principal, self.attendant = principal, attendant

    def run(self, first_speaker=None, n_turns=1):
        # assign first speaker and listener
        speaker, listener = self.principal, self.attendant
        if first_speaker == "a":
            speaker, listener = listener, speaker

        message = Message(type='start')
        for n in range(n_turns):
            message = speaker.speak(message)
            listener.listen(message)
            # swap role for the next turn
            speaker, listener = listener, speaker

    def eval(self, executor_role=None):
        executor = self.principal if executor_role == "a" else self.attendant

        plan = executor.plan()
        is_solved = self.principal.verify(plan)

        return plan, is_solved




