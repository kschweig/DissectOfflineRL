import os
import gym
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from source.evaluation import evaluate
from source.utils import get_agent


def train_offline(experiment, envid, agent_type="DQN", transitions=200000, batch_size=128, use_run=1, run=1, seed=42,
                  use_subset=False, lower=None, upper=None,
                  use_progression=False, buffer_size=None,
                  use_sim=False,
                  use_density=False):

    # over how many episodes do we take average and how much gradient updates to next
    mean_over = 100
    evaluate_every = 50

    env = gym.make(envid)
    obs_space = len(env.observation_space.high)
    agent = get_agent(agent_type, obs_space, env.action_space.n, seed)

    # load saved buffer
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{use_run}.pkl"), "rb") as f:
        buffer = pickle.load(f)

    # configure buffer
    buffer.batch_size = batch_size

    #######################
    # experiment specific #
    #######################

    if use_subset: buffer.subset(lower, upper)
    if use_sim: buffer.calc_sim()
    if use_density: buffer.calc_density()

    #######################

    # seeding
    env.seed(seed)
    buffer.set_seed(seed)
    torch.manual_seed(seed)

    writer = SummaryWriter(log_dir=os.path.join("runs", f"ex{experiment}", f"{envid}_{agent_type}_run{run}"))

    all_rewards = []

    for iter in tqdm(range(transitions), desc=f"{agent_type} ({envid}), run {run}"):
        if use_progression:
            minimum = max(0, iter - buffer_size)
            maximum = max(batch_size, iter)
        else:
            minimum = None
            maximum = None

        agent.train(buffer, writer, maximum, minimum, (use_density or use_sim))

        if (iter+1) % evaluate_every == 0:
            all_rewards = evaluate(env, agent, writer, all_rewards, over_episodes=mean_over)

    # mean rewards
    mean_rewards = []
    for i in range(1, len(all_rewards)):
        from_ = max(0, i-mean_over)
        mean_rewards.append(np.mean(all_rewards[from_:i]))

    return agent


# test training
if __name__ == "__main__":

    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
    envid = envs[2]
    agent_type = "REM"

    print("solving ", envid, " offline with ", agent_type)

    agent = train_offline(1, envid, agent_type, transitions=200000, batch_size=128, run=2, use_run=1)

    # showcase policy
    env = gym.make(envid)
    state, done, iteration = env.reset(), False, 0
    while not done and iteration < 200:
        env.render()
        action, _, _ = agent.policy(state)
        state, _, done, _ = env.step(action)
        iteration += 1
    env.close()