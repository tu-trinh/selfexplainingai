import os
import sys


class Speaker(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def speak(self, message):
        assert message.type == "trajectory_description"
        # TODO: sample a set of edits
        edits = None
        return edits


class Listener(Agent):

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def speak(self, message):
        assert message.type == "start"
        # return a dummy description
        return Message(type="trajectory_description", content=None)

    def listen(self, message):
        assert message.type == "belief_mismatch"
        self.env = LGWM(base_env=self.env, edits=message.content, model=model)

    def plan(self):
        plan = OraclePlanner(self.env)
        return plan
