from constants import *
from llm import LLM
from utils import *
import gymnasium
from envs.enums import *
from typing import Callable, Dict, List, Tuple, Any, Union
from trajectories import *


class Agent:
    def __init__(self, name: str, query_source: str, model_source: str):
        self.name = name
        self.world_model = None
        self.skills = []
        self.rewards_and_weights = []

        self.llm = LLM(query_source, model_source)
        self.tokens = 0
        self.additional_actions = None

        self.interactions = 0

    
    def set_world_model(self, env: gymnasium.Env) -> None:
        self.world_model = env
    
    
    def add_skill(self, skill: Skill) -> None:
        self.skills.append(skill)
    
    
    def set_reward_function(self, reward_function: Callable[[Dict], float], weight: float) -> None:
        self.rewards_and_weights.append((reward_function, weight))
    

    def get_reward(self, env_state: Dict) -> float:
        return sum([w * rf(env_state) for rf, w in self.rewards_and_weights])
    

    def get_action(self, utterance: Union[str, Dict]) -> Skill:
        if isinstance(utterance, str):
            skill = self.llm.match(utterance)
            return ACTION_TO_IDX[skill]
        else:
            pass  # see get_action below in LearningAgent
    

    def act(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        items = self.world_model.step(action)
        return items

    
    def speak(self, utterance: Any) -> Any:
        raise NotImplementedError("Must be implemented by all agents")
    

    def listen(self, utterance: Any) -> Any:
        raise NotImplementedError("Must be implemented by all agents")


class TeachingAgent(Agent):
    def __init__(self, name, query_source, model_source):
        super().__init__(name, query_source, model_source)
    

    def speak(self, skills: List[Skill] = None, world_model: gymnasium.Env = None):
        assert (not skills and not world_model) or (skills and world_model), "Principal must either receive nothing or both skills and world model before speaking"
        pass


    def listen(self, trajectory: Trajectory) -> bool:
        pass


class LearningAgent(Agent):
    def __init__(self, name, query_source, model_source):
        super().__init__(name, query_source, model_source)

    def set_instruction(self, instruction, additional_actions = None):
        self.llm.set_instruction(instruction, additional_actions)
        self.instruction = None
        self.additional_actions = additional_actions
    
    def get_action(self, observation = None, action_failed = False):
        response = self.llm.get_action(observation, action_failed)
        parsed_response = convert_response_to_action(response, self.additional_actions)
        self.interactions += 1
        self.llm.responses.append(parsed_response)
        self.tokens = self.llm.total_prompt_tokens
        return parsed_response
    

    def speak(self, utterance: str = None, trajectories: List[Trajectory] = None) -> Trajectory:
        assert 


    def listen(self, utterance):
        pass


if __name__ == "__main__":
    la = LearningAgent("Taniqua", "openai", "gpt-3.5-turbo")