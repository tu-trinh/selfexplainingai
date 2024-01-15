class Transition:
    def __init__(self,
                 obs: Dict,
                 action: int,
                 reward: float,
                 next_obs: Dict,
                 terminated: bool = None,
                 truncated: bool = None,
                 info: Dict = None):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.terminated = terminated
        self.truncated = truncated
        self.info = info


class Trajectory:
    def __init__(self):
        self.transitions = []

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)
    
    def __iter__(self):
        return iter(self.transitions)
