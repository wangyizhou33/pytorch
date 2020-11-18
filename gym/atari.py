import gym

# env = gym.make('SpaceInvaders-v0')
env = gym.make("Enduro-v0")

env.reset()
while True:
    env.render()