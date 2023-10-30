from abc import ABC, abstractmethod
import openai
import numpy as np
import torch
from constants import *

class Agent(ABC):
    pass

class TeachingAgent(Agent):
    pass

class LearningAgent(Agent):
    def __init__(self, name, source, model):
        self.name = name
        self.source = source
        if self.source == "openai":
            openai.api_key = OPENAI_KEY
        self.model = model
        self.system_message = None
        self.prompts = []
        self.responses = []
        self.rounds = 0
        self.interactions = 0
    
    def set_instruction(self, instruction):
        self.system_message = {"role": "system", "content": "You are an agent who is trying to complete a task in an unknown environment. Your abilities include moving forward, turning left, turning right, picking things up, and using things on other things. Your task is: " + instruction}
        self.prompts = []
        self.responses = []
        self.rounds += 1
        self.interactions = 0
    
    def get_action(self, observation = None):
        if self.source == "openai":
            messages = [self.system_message]
            for i in range(self.interactions):
                try:
                    messages.append({"role": "user", "content": self.prompts[i]})
                    messages.append({"role": "assistant", "content": self.responses[i]})
                except IndexError:
                    break
            if observation:
                complete_instruction = observation + " " + INQUIRY
                messages.append({"role": "user", "content": complete_instruction})
            else:
                self.responses.pop()
                messages.pop()
            response_obj = openai.ChatCompletion.create(
                model = self.model,
                messages = messages,
                temperature = TEMPERATURE
            )
            response = response_obj["choices"][0]["message"]["content"]
        if observation:
            self.prompts.append(complete_instruction)
        self.responses.append(response)
        self.interactions += 1
        return response
    
    def display_history(self):
        messages = [self.system_message]
        for i in range(self.interactions):
            try:
                messages.append({"role": "user", "content": self.prompts[i]})
                messages.append({"role": "assistant", "content": self.responses[i]})
            except IndexError:
                break
        return messages

if __name__ == "__main__":
    la = LearningAgent("Taniqua", "openai", "gpt-3.5-turbo")