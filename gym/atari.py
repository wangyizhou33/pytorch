import gym

env = gym.make('SpaceInvaders-v0')
# env = gym.make("Enduro-v0")
env.reset()

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()