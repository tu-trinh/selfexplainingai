from abc import ABC, abstractmethod
from constants import *
from llm import LLM
from utils import *


class Agent(ABC):
    pass

class TeachingAgent(Agent):
    pass

class LearningAgent(Agent):
    def __init__(self, name, query_source, model_source):
        self.name = name
        self.llm = LLM(query_source, model_source)
        self.interactions = 0
        self.tokens = 0
        self.instructions = None
        self.additional_actions = None
    
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


if __name__ == "__main__":
    la = LearningAgent("Taniqua", "openai", "gpt-3.5-turbo")