import llmengine
from ..constants import *


prompt = """Your goal is: go to the green ball
Observation: You see a wall 2 step left, You see a purple key 1 step left
and 2 steps forward, You see a yellow key 1 step left and 1 step forward,
You see a green ball 3 steps forward, You see a grey ball 1 step right and
5 steps forward, You see a green key 1 step right and 2 steps forward,
You see a grey ball 1 step right and 1 step forward, You see a green key
2 steps right and 4 steps forward, You see a red box 2 steps right and
2 steps forward.
What action should you take?"""
llmengine.api_engine.api_key = SCALE_KEY
response_obj = llmengine.Completion.create(
    prompt = prompt,
    model = "mistral-7b-instruct",
    temperature = TEMPERATURE,
    max_new_tokens = MAX_NEW_TOKENS,
    timeout = 120
)
response = response_obj.output.text
print("***")
print(response)
print("***")