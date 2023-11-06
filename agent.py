from abc import ABC, abstractmethod
import openai
import numpy as np
import torch
import constants
from constants import *
import tiktoken

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
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.system_message = None
        self.prompts = []
        self.responses = []
        self.rounds = 0
        self.interactions = 0
        self.instruction = None
    
    def set_instruction(self, instruction):
        self.system_message = {"role": "system", "content": SETTING_DESCRIPTION + "\nYOUR TASK IS: " + instruction}
        self.instruction = instruction
        self.prompts = []
        self.responses = []
        self.rounds += 1
        self.interactions = 0
    
    def get_action(self, observation = None, action_failed = False):
        if self.source == "openai":
            messages = [self.system_message]
            for i in range(self.interactions):
                try:
                    messages.append({"role": "user", "content": self.prompts[i]})
                    messages.append({"role": "assistant", "content": self.responses[i]})
                except IndexError:
                    break
            if observation:
                if action_failed:
                    complete_instruction = "The previous action you did achieved nothing. "
                else:
                    complete_instruction = ""
                complete_instruction += observation + "Reminder that your task is, " + self.instruction + " " + INQUIRY
                messages.append({"role": "user", "content": complete_instruction})
            else:
                self.responses.pop()
                messages.pop()
            messages = self.clean_messages(messages)
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
    
    def clean_messages(self, messages):
        content_length = 0
        for message in messages:
            content_length += len(self.tokenizer.encode(message["content"]))
        while content_length > constants.MAX_MSG_TOKENS:
            messages = messages[:1] + messages[3:]  # keep system message, throw out the oldest user/assistant pair
            content_length = 0
            for message in messages:
                token_count = len(self.tokenizer.encode(message["content"]))
                content_length += token_count
        return messages
    
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