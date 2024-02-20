import os
import sys


class LanguageAttendant(Agent):

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def speak(self, message):
        assert message.type == "trajectory_description"
        # message is a description of a trajectory
        prediction = model.predict(message.content)
        return prediction


class LanguagePrincipal(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def speak(self, message):
        assert message.type == 'start'
        # return a description of a trajectory
        plan = OraclePlanner(self.env)
        description = self.verbalize_plan(plan)
        return Message(type="trajectory_description", content=description)

    def listen(self, message):
        assert message == "belief_mismatch"
        self.env = make_env(self.env, message.content)

    def plan(self):
        plan = OraclePlanner(new_env)
        return plan

