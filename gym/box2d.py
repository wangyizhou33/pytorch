import gym  # openAi gym
from gym import envs

env = gym.make("CarRacing-v0")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()