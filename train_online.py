import os
import gym
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from source.buffer import ReplayBuffer
from source.agents.dqn import DQN

seed = 42
batch_size = 32
buffer_size = 20000
transitions = 20000
show_every = 5
train_every = 1
train_start_iter = batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')
obs_space = len(env.observation_space.high)
agent = DQN(obs_space, env.action_space.n, seed=seed)
buffer = ReplayBuffer(obs_space, buffer_size, batch_size, seed=seed)
savebuffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)

# seeding
env.seed(seed)
eval_env.seed(seed)
torch.manual_seed(seed)

all_rewards = []
rewards = []
ep_reward = 0
values = []
done = True
ep = 0

for iter in tqdm(range(transitions)):
    # Reset if environment is done
    if done:
        state = env.reset()
        rewards.append(ep_reward)
        ep_reward = 0
        ep += 1


    # obtain action
    action, value = agent.policy(state)

    values.append(value)

    # step in environment
    next_state, reward, done, _ = env.step(action)

    # add to buffer
    buffer.add(state, action, reward, done, next_state)
    savebuffer.add(state, action, reward, done, next_state)

    ep_reward += reward

    # now next state is the new state
    state = next_state

    # update the agent periodically after initially populating the buffer
    if (iter+1) % train_every == 0 and iter > train_start_iter:
        agent.train(buffer)

    if (iter+1) % show_every == 0:
        done_ = False
        state_ = eval_env.reset()
        while not done_:
            action_, value_ = agent.policy(state_, eval=True)
            state_, reward_, done_, _ = eval_env.step(action_)
            # state = buffer.get_closest(state)
            ep_reward += reward_
            values.append(value_)

        all_rewards.append(ep_reward)
        print(f"cur_reward: ", ep_reward, "| mean reward: ", round(np.mean(all_rewards[-100:]), 2),
              " | values: ", round(np.nanmean(values), 2))

        ep_reward = 0
        values = []

# save buffer for offline training
with open(os.path.join("data", "buffer.pkl"), "wb") as f:
    pickle.dump(savebuffer, f)

# showcase policy
state, done = env.reset(), False
while not done and iter < 500:
    env.render()
    action, _ = agent.policy(state)
    state, _, done, _ = env.step(action)
env.close()

# mean rewards
mean_rewards = []
for i in range(len(all_rewards)):
    from_ = max(0, i-100)
    mean_rewards.append(np.mean(all_rewards[from_:i]))

# plot learning
plt.figure(figsize=(8, 6))
plt.ylabel("Reward")
plt.xlabel("Episodes")
plt.plot(range(len(all_rewards)), all_rewards, label="reward")
plt.plot(range(len(mean_rewards)), mean_rewards, label="mean reward")
plt.legend()
plt.show()