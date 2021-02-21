import gym
import gym_super_mario_bros

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env.reset()

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()