import os
import torch
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils.buffer import ReplayBuffer
from .utils.evaluation import evaluate
from .utils.utils import get_agent, make_env


def train_online(experiment, agent_type="DQN", discount=0.95, envid='CartPole-v1', transitions=200000, buffer_size=50000, run=1, seed=42):

    batch_size = 32
    evaluate_every = 100
    mean_over = 100
    train_every = 1
    train_start_iter = batch_size

    writer = SummaryWriter(log_dir=os.path.join("runs", f"ex{experiment}", f"{envid}_online_run{run}"))

    env = make_env(envid)
    eval_env = make_env(envid)
    obs_space = len(env.observation_space.high)

    agent = get_agent(agent_type, obs_space, env.action_space.n, discount, seed)

    # two buffers, one for learning, one for storing all transitions!
    buffer = ReplayBuffer(obs_space, buffer_size, batch_size, seed=seed)
    er_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)

    # seeding
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ep_rewards = []
    all_rewards = []
    done = True
    ep = 0

    for iteration in tqdm(range(transitions), desc=f"Behavioral policy ({envid}), run {run}"):
        # Reset if environment is done
        if done:
            if iteration > 0:
                ep_rewards.append(ep_reward)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    writer.add_scalar("train/Reward (SMA)", np.mean(ep_rewards[-100:]), ep)
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
        er_buffer.add(state, action, reward, done, next_state)

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
    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}.pkl"), "wb") as f:
        pickle.dump(er_buffer, f)

    # mean rewards
    mean_rewards = []
    for i in range(1, len(all_rewards)):
        from_ = max(0, i-mean_over)
        mean_rewards.append(np.mean(all_rewards[from_:i]))

    return agent