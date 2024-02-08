import package.builder as builder
from package.builder import *
import matplotlib.pyplot as plt

config_file = "package/configs/test1_difficulty1.yaml"

principal, attendant = builder.make_agents(config_file)
builder.set_advanced_reward_functions(config_file, principal, attendant)  # optional

agent = attendant
agent.world_model.reset()
while True:
    action = agent.world_model.action_space.sample()
    agent.act(action)
    agent.world_model.render()

image = principal.speak(mode = "image")
differences = attendant.listen(image = image)
adapted_solution = principal.listen(differences)
trajectory = attendant.execute_actions(adapted_solution)
is_solved = principal.verify(trajectory)
