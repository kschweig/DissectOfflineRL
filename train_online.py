import os
import gym
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from source.buffer import ReplayBuffer
from source.evaluation import evaluate
from source.agents.dqn import DQN


def train_online(experiment, envid='CartPole-v1', run=1):

    seed = 42
    batch_size = 32
    buffer_size = 50000
    transitions = 200000
    evaluate_every = 50
    mean_over = 100
    train_every = 1
    train_start_iter = batch_size

    writer = SummaryWriter(log_dir=os.path.join("runs", f"ex{experiment}_{envid}_online_run{run}"))

    env = gym.make(envid)
    eval_env = copy.deepcopy(env)
    obs_space = len(env.observation_space.high)
    agent = DQN(obs_space, env.action_space.n, seed=seed)

    # two buffers, one for learning, one for storing all transitions!
    buffer = ReplayBuffer(obs_space, buffer_size, batch_size, seed=seed)
    savebuffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)

    # seeding
    env.seed(seed)
    eval_env.seed(seed)
    torch.manual_seed(seed)

    all_rewards = []
    done = True
    ep = 0

    for iteration in tqdm(range(transitions)):
        # Reset if environment is done
        if done:
            if iteration > 0:
                writer.add_scalar("train/Reward", ep_reward, ep)
                writer.add_scalar("train/Values", np.nanmean(values), ep)
                writer.add_scalar("train/Entropy", np.nanmean(entropies), ep)

            state = env.reset()
            ep_reward, values, entropies = 0, [], []
            ep += 1

        # obtain action
        action, value , entropy = agent.policy(state)

        # step in environment
        next_state, reward, done, _ = env.step(action)

        # add to buffer
        buffer.add(state, action, reward, done, next_state)
        savebuffer.add(state, action, reward, done, next_state)

        # add reward, value and entropy of current step for means over episode
        ep_reward += reward
        values.append(value)
        entropies.append(entropy)

        # now next state is the new state
        state = next_state

        # update the agent periodically after initially populating the buffer
        if (iteration+1) % train_every == 0 and iteration > train_start_iter:
            agent.train(buffer, writer)

        # test agent on environment if executed greedily
        if (iteration+1) % evaluate_every == 0:
            all_rewards = evaluate(eval_env, agent, writer, all_rewards, over_episodes=mean_over)

    # save buffer for offline training
    with open(os.path.join("data", f"ex{experiment}_{envid}_run{run}.pkl"), "wb") as f:
        pickle.dump(savebuffer, f)

    # mean rewards
    mean_rewards = []
    for i in range(len(all_rewards)):
        from_ = max(0, i-mean_over)
        mean_rewards.append(np.mean(all_rewards[from_:i]))

    return agent


# test training
if __name__ == "__main__":

    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
    envid = envs[1]
    print("Solving ", envid, " online.")

    agent = train_online(99, envid, run=99)

    # showcase policy
    env = gym.make(envid)
    state, done, iteration = env.reset(), False, 0
    while not done and iteration < 200:
        env.render()
        action, _, _ = agent.policy(state)
        state, _, done, _ = env.step(action)
        iteration += 1
    env.close()
