import gym

env = gym.make('CartPole-v0')
obs = env.reset()
print(obs)
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()