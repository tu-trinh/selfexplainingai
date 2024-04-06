from package.infrastructure.access_tokens import *
from package.infrastructure.llm_constants import *
from package.infrastructure.basic_utils import debug

import openai
import llmengine
from collections import deque
from transformers import AutoTokenizer
import tiktoken
import requests


AVAILABLE_QUERY_SOURCES = ["openai", "huggingface", "scale"]
AVAILABLE_MODEL_SOURCES = ["gpt", "mistral"]
MODEL_MAPPING = {
    "gpt": {"openai": "gpt-4-1106-preview"},
    "mistral": {"scale": "mistral-7b-instruct", "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"},
    "mixtral": {"scale": "mixtral-8x7b-instruct", "huggingface": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
}
TOKENIZER_MAPPING = {
    "gpt": "gpt-4-1106-preview",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}
assert all([src in MODEL_MAPPING for src in AVAILABLE_MODEL_SOURCES]), "Missing a model mapping"


class LLM:
    def __init__(self, query_source, model_source):
        self.query_source = query_source if query_source else "openai"
        self.model_source = model_source if model_source else "gpt"
        try:
            self.model = MODEL_MAPPING[self.model_source][self.query_source]
        except:
            raise AssertionError("No matching model name for model source and query source combo")
        if self.query_source == "openai":
            openai.api_key = OPENAI_KEY
            self.tokenizer = tiktoken.encoding_for_model(TOKENIZER_MAPPING[self.model_source])
        elif self.query_source == "scale":
            llmengine.api_engine.api_key = SCALE_KEY
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAPPING[self.model_source])
        elif self.query_source == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAPPING[self.model_source])
        
        self.system_message = None
        self.prompts = deque()
        self.responses = deque()
        self.total_prompt_tokens = 0

    
    def count_tokens(self, strings):
        assert isinstance(strings, list), "Must pass in list of strings for token counting"
        for string in strings:
            if self.query_source == "openai":
                self.total_prompt_tokens += len(self.tokenizer.encode(string))
            else:
                self.total_prompt_tokens += len(self.tokenizer.tokenize(string))
    
    
    def set_instruction(self, instruction, additional_actions):
        if additional_actions:
            self.system_message = SETTING_DESCRIPTION + additional_actions + TASK_PROLOGUE + instruction + "\n"
        else:
            self.system_message = SETTING_DESCRIPTION + TASK_PROLOGUE + instruction + "\n"
    
    
    def get_formatted_prompts_and_responses(self):
        if len(self.prompts) == 1:
            return [f"Current timestep: {self.prompts[0]}\n{INQUIRY}"]
        if len(self.prompts) == 2:
            return [
                f"One timestep ago: {self.prompts[0]}",
                f"Your chosen action: {self.responses[0]}",
                f"Current timestep: {self.prompts[1]}\n{INQUIRY}"
            ]
        return [
            f"Two timesteps ago: {self.prompts[0]}",
            f"Your chosen action: {self.responses[0]}",
            f"One timestep ago: {self.prompts[1]}",
            f"Your chosen action: {self.responses[1]}",
            f"Current timestep: {self.prompts[2]}\n{INQUIRY}"
        ]
    
    
    def get_action(self, observation, action_failed):
        assert self.system_message is not None, "System message not yet set"

        if observation:
            if action_failed:
                this_prompt = "The previous action you attempted failed. "
            else:
                this_prompt = ""
            this_prompt += observation
            if len(self.prompts) > PROMPT_HISTORY_LIMIT:
                self.prompts.popleft()
            self.prompts.append(this_prompt)
            if len(self.responses) > PROMPT_HISTORY_LIMIT - 1:
                self.responses.popleft()
        formatted_prompts_and_responses = self.get_formatted_prompts_and_responses()

        if self.query_source == "openai":
            if len(self.prompts) > 1:
                system_message = self.system_message + PROMPT_FORMAT_INSTRUCTION
            else:
                system_message = self.system_message
            messages = [{"role": "system", "content": system_message}]
            for i in range(len(formatted_prompts_and_responses)):
                if i % 2 == 0:
                    messages.append({"role": "user", "content": formatted_prompts_and_responses[i]})
                else:
                    messages.append({"role": "assistant", "content": formatted_prompts_and_responses[i]})
            try:
                response_obj = openai.ChatCompletion.create(
                    model = self.model,
                    messages = messages,
                    temperature = TEMPERATURE
                )
                response = response_obj["choices"][0]["message"]["content"]
            except Exception as e:
                print("Could not complete LLM request due to", e)
            self.count_tokens([system_message] + formatted_prompts_and_responses)
        
        elif self.query_source == "scale":
            prompt = self.system_message
            if len(self.prompts) > 1:
                prompt += PROMPT_FORMAT_INSTRUCTION
            else:
                prompt += "\n"
            prompt += "\n".join(formatted_prompts_and_responses)
            try:
                response_obj = llmengine.Completion.create(
                    prompt = prompt,
                    model = self.model,
                    temperature = TEMPERATURE,
                    max_new_tokens = MAX_NEW_TOKENS,
                    timeout = 120
                )
                response = response_obj.output.text
            except Exception as e:
                print("Could not complete LLM request due to", e)
            self.count_tokens([prompt])
        
        elif self.query_source == "huggingface":
            prompt = self.system_message
            if len(self.prompts) > 1:
                prompt += PROMPT_FORMAT_INSTRUCTION
            else:
                prompt += "\n"
            prompt += "\n".join(formatted_prompts_and_responses)
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
            payload = {
                "inputs": prompt,
                "options": {"wait_for_model": True},
                "parameters": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}
            }
            try:
                response_obj = requests.post(url, headers = headers, json = payload)
                response = response_obj.json()[0]["generated_text"][len(prompt):]
            except Exception as e:
                print("Could not complete LLM request due to", e)
            self.count_tokens([prompt])
        
        return response
    

    def get_differences(self, other_desc, own_desc):
        prompt = DIFFERENCES_QUESTION.format(other_env_desc = other_desc,
                                             own_env_desc = own_desc,
                                             differences = "\n".join([f"({k}) {v}" for k, v in DIFFERENCES_MAPPING]))
        if self.query_source == "openai":
            try:
                response_obj = openai.ChatCompletion.create(
                    model = self.model,
                    messages = [{"role": "user", "content": prompt}],
                    temperature = TEMPERATURE
                )
                response = response_obj["choices"][0]["message"]["content"]
            except Exception as e:
                print("Could not complete LLM request due to", e)
        differences = []
        for diff_letter in response.split(","):
            differences.append(DIFFERENCES_MAPPING[diff_letter])
        return ". ".join(differences)
    
    def get_skill_description(self, obs_act_seq, available_skills):
        with open("package/configs/skills.txt", "r") as f:
            skill_choices = [skill.strip() for skill in f.readlines()]
        prompt = GET_SKILL_NAME_QUESTION.format(obs_act_seq = obs_act_seq, skill_choices = "\n".join(available_skills)).strip()
        debug(prompt)
        if self.query_source == "openai":
            try:
                response_obj = openai.ChatCompletion.create(
                    model = self.model,
                    messages = [{"role": "user", "content": prompt}],
                    temperature = TEMPERATURE
                )
                response = response_obj["choices"][0]["message"]["content"]
            except Exception as e:
                print("Could not complete LLM request due to", e)
                response = None
            return response
    

    def get_new_plan_based_on_skills(self, task, obs_act_seq, skills):
        prompt = GET_NEW_PLAN_BASED_ON_SKILLS_QUESTION.format(task = task, obs_act_seq = obs_act_seq, skill_choices = "\n".join(skills)).strip()
        debug("Asked the model:")
        debug(prompt)
        if self.query_source == "openai":
            try:
                response_obj = openai.ChatCompletion.create(
                    model = self.model,
                    messages = [{"role": "user", "content": prompt}],
                    temperature = TEMPERATURE
                )
                response = response_obj["choices"][0]["message"]["content"]
            except Exception as e:
                print("Could not complete LLM request due to", e)
                response = None
            return response
