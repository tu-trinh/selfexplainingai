import babyai_text
import gym
import gymnasium

env = gym.make("BabyAI-MixedTrainLocal")
obs, info = env.reset()
print(obs)
print(info)

# np bool is not supported
# env = gym.make("MiniGrid-Empty-5x5-v0")
# ret = env.reset()
# print(ret)

# doesn't exist
# env = gymnasium.make("BabyAI-MixedTrainLocal")
# ret = env.reset()
# print(ret)

# doesn't exist
# env = gymnasium.make("MiniGrid-Empty-5x5-v0")
# ret = env.reset()
# print(ret)

# gymnasium.pprint_registry()