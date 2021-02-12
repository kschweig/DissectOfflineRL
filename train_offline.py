import os
import gym
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from source.buffer import ReplayBuffer
from source.agents.dqn import DQN

seed = 42
batch_size = 20
buffer_size = 100000
transitions = buffer_size
show_every = 2000
train_every = 1
train_start_iter = batch_size

env = gym.make('CartPole-v1')
obs_space = len(env.observation_space.high)
agent = DQN(obs_space, env.action_space.n, seed=seed)
buffer = ReplayBuffer(obs_space, buffer_size, batch_size, seed=seed)
# load saved buffer
with open(os.path.join("data", "buffer.pkl"), "rb") as f:
    buffer = pickle.load(f)
    buffer.calc_sim()

all_rewards = []
ep_reward = 0
values = []

for iter in tqdm(range(transitions)):

    agent.train(buffer)

    if (iter+1) % show_every == 0:
        done = False
        state = env.reset()
        while not done:
            action, value = agent.policy(state)
            state, reward, done, _ = env.step(action)
            state = buffer.get_closest(state)
            ep_reward += reward
            values.append(value)

        print(f"reward: ", ep_reward, " | values: ", round(np.nanmean(values), 2))

        all_rewards.append(ep_reward)
        ep_reward = 0
        values = []


plt.figure(figsize=(8, 6))
plt.ylabel("Reward")
plt.xlabel("Episodes")
plt.plot(range(len(all_rewards)), all_rewards)
plt.show()