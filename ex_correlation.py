import os
import torch
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from source.utils.buffer import ReplayBuffer
from source.utils.evaluation import evaluate
from source.offline_ds_evaluation.evaluator import Evaluator
from source.utils.utils import get_agent, make_env
from gym_minigrid.wrappers import ReseedWrapper


# keep training parameters for online training fixed, the experiment does not interfere here.
seed = 42
seeds = [seed]
#seeds = [seed, seed+100, seed+200, seed+300, seed+400]
batch_size = 32
buffer_size = 50000
transitions = buffer_size
lr = 1e-4
evaluate_every = 100
mean_over = 100
train_every = 1
train_start_iter = batch_size

writer = SummaryWriter(log_dir=os.path.join("runs", "ex_corr", "MiniGrid-LavaGapS7-v0", "online", "DQN",
                                            f"{len(seeds)}_seeds"))
env = make_env("MiniGrid-LavaGapS7-v0")
eval_env = make_env("MiniGrid-LavaGapS7-v0")
if len(seeds) > 0:
    env = ReseedWrapper(env, seeds=seeds)
    eval_env = ReseedWrapper(eval_env, seeds=seeds)

obs_space = len(env.observation_space.high)

agent = get_agent("DQN", obs_space, env.action_space.n, 0.95, lr, seed)

# two buffers, one for learning, one for storing all transitions!
probas = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
er_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_0 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_01 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_05 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_10 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_25 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_50 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_75 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
test_100 = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)

# seeding
env.seed(seed)
eval_env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

ep_rewards, all_rewards, all_dev_mean, all_dev_std = [], [], [], []
done = True
ep = 0
"""
#####################################
# train agent
#####################################
for iteration in tqdm(range(transitions), desc=f"Behavioral policy"):
    # Reset if environment is done
    if done:
        if iteration > 0:
            ep_rewards.append(ep_reward)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                writer.add_scalar("train/Reward (SMA)", np.mean(ep_rewards[-100:]), ep)
                writer.add_scalar("train/Reward", ep_reward, ep)
                writer.add_scalar("train/Max-Action-Value (mean)", np.nanmean(action_values), ep)
                writer.add_scalar("train/Max-Action-Value (std)", np.nanstd(action_values), ep)
                writer.add_scalar("train/Values", np.nanmean(values), ep)
                writer.add_scalar("train/Action-Values std", np.nanmean(values_std), ep)
                writer.add_scalar("train/Entropy", np.nanmean(entropies), ep)

        state = env.reset()
        ep_reward, values, values_std, action_values, entropies = 0, [], [], [], []
        ep += 1

    # obtain action
    action, value, entropy = agent.policy(state)

    # step in environment
    next_state, reward, done, _ = env.step(action)

    # add to buffer
    er_buffer.add(state, action, reward, done)

    # add reward, value and entropy of current step for means over episode
    ep_reward += reward
    try:
        values.append(value.numpy().mean())
        values_std.append(value.numpy().std())
        action_values.append(value.max().item())
    except AttributeError:
        values.append(value)
        values_std.append(value)
        action_values.append(value)
    entropies.append(entropy)

    # now next state is the new state
    state = next_state

    # update the agent periodically after initially populating the buffer
    if (iteration+1) % train_every == 0 and iteration > train_start_iter:
        agent.train(er_buffer, writer)

    # test agent on environment if executed greedily
    if (iteration+1) % evaluate_every == 0:
        all_rewards, all_dev_mean, all_dev_std = evaluate(eval_env, agent, writer, all_rewards,
                                                          all_dev_mean, all_dev_std, over_episodes=mean_over)

# save ER-buffer for further processing
os.makedirs(os.path.join("data", "ex_corr", f"{len(seeds)}_seeds"), exist_ok=True)
with open(os.path.join("data", "ex_corr", f"{len(seeds)}_seeds", f"er_buffer.pkl"), "wb") as f:
    pickle.dump(er_buffer, f)

for buffer, eps in zip([test_0, test_01, test_05, test_10, test_25, test_50, test_75, test_100], probas):
    done, n_actions = True, env.action_space.n
    agent.eval_eps = eps
    for _ in tqdm(range(transitions), desc=f"Evaluate {eps}-greedy policy"):
        if done:
            state = env.reset()

        action, _, _ = agent.policy(state, eval=True)
        next_state, reward, done, _ = env.step(action)

        buffer.add(state, action, reward, done)

        state = next_state

    os.makedirs(os.path.join("data", "ex_corr", f"{len(seeds)}_seeds"), exist_ok=True)
    with open(os.path.join("data", "ex_corr", f"{len(seeds)}_seeds", f"test_{int(eps*100)}.pkl"), "wb") as f:
        pickle.dump(buffer, f)

"""

names = ["er_buffer", "test_0", "test_1", "test_5", "test_10", "test_25", "test_50", "test_75", "test_100"]
rewards, entropies, randomness = [], [], []
unique_states, unique_state_actions, state_coverages, state_action_coverages = [], [], [], []

for name in names:
    with open(os.path.join("data", "ex_corr", f"{len(seeds)}_seeds", name + ".pkl"), "rb") as f:
        buffer = pickle.load(f)

    evaluator = Evaluator("MiniGrid-LavaGapS7-v0", name, buffer.state, buffer.action, buffer.reward,
                          np.invert(buffer.not_done))

    evaluator.train_state_embedding(epochs=10)
    evaluator.train_state_action_embedding(epochs=10)
    evaluator.train_behavior_policy(epochs=5)

    evaluator.plot_states()
    evaluator.plot_state_actions()

    rewards.append(np.mean(evaluator.get_rewards()) / 0.95)
    entropy = np.mean(evaluator.get_bc_entropy())
    entropies.append(entropy)

    unique_states.append(evaluator.get_unique_states())
    unique_state_actions.append(evaluator.get_unique_state_actions())
    state_coverages.append(evaluator.get_state_pseudo_coverage())
    state_action_coverages.append(evaluator.get_state_action_pseudo_coverage())
    
    print(name, rewards[-1], entropies[-1], unique_states[-1], state_coverages[-1], unique_state_actions[-1], state_action_coverages[-1])

# plotting
f, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].plot(probas, entropies[1:], "-o", color="C0", label="entropy")
axs[0].plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), linestyle="dotted", color="black")
axs[0].set_xlabel("$\epsilon$")
axs[0].set_ylabel("Entropy")

xmin = max(np.min(unique_states), np.min(state_coverages) * 100**2)
xmax = min(np.max(unique_states), np.max(state_coverages) * 100**2)
ymin = max(np.min(unique_states) / 100**2, np.min(state_coverages))
ymax = min(np.max(unique_states) / 100**2, np.max(state_coverages))
axs[1].plot(unique_states[1:], state_coverages[1:], "-o", color="C1", label="state coverage")
axs[1].plot(np.arange(xmin, xmax, (xmax - xmin) / 100), np.arange(ymin, ymax, (ymax - ymin) / 100),
            linestyle="dotted", color="black")
axs[1].set_xlabel("Unique States")
axs[1].set_ylabel("Pseudo-State-Coverage")
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

xmin = max(np.min(unique_state_actions), np.min(state_action_coverages) * 100**2)
xmax = min(np.max(unique_state_actions), np.max(state_action_coverages) * 100**2)
ymin = max(np.min(unique_state_actions) / 100**2, np.min(state_action_coverages))
ymax = min(np.max(unique_state_actions) / 100**2, np.max(state_action_coverages))
axs[2].plot(unique_state_actions[1:], state_action_coverages[1:], "-o", color="C2", label="state-action coverage")
axs[2].plot(np.arange(xmin, xmax, (xmax - xmin) / 100), np.arange(ymin, ymax, (ymax - ymin) / 100),
            linestyle="dotted", color="black")
axs[2].set_xlabel("Unique State-Actions")
axs[2].set_ylabel("Pseudo-State-Action-Coverage")
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

f.text(0.5, 0.98, "1 seed" if len(seeds) == 1 else f"{len(seeds)} seeds", ha='center', fontsize="large")
f.tight_layout()
plt.show()