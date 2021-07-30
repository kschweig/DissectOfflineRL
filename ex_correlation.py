import os
import torch
import pickle
import warnings
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from source.utils.buffer import ReplayBuffer
from source.utils.evaluation import evaluate
from source.offline_ds_evaluation.evaluator import Evaluator
from source.utils.utils import get_agent, make_env
from gym_minigrid.wrappers import ReseedWrapper

test_env = "MountainCar-v0"#"MiniGrid-LavaGapS7-v0"#

# keep training parameters for online training fixed, the experiment does not interfere here.
seed = 42
seed_list = [[seed],
             [seed, seed+100, seed+200],
             [seed, seed+100, seed+200, seed+300, seed+400],
             []]

# no loop to make it runnable by multiple processes
seeds = seed_list[3]

embedding_epochs = 10
batch_size = 32
buffer_size = 50000
transitions = buffer_size
lr = 1e-4
evaluate_every = 100
mean_over = 100
train_every = 1
train_start_iter = batch_size

writer = SummaryWriter(log_dir=os.path.join("runs", "ex_corr", test_env, "online", "DQN",
                                            f"{len(seeds)}_seeds"))
env = make_env(test_env)
eval_env = make_env(test_env)
if len(seeds) > 0:
    env = ReseedWrapper(env, seeds=seeds)
    eval_env = ReseedWrapper(eval_env, seeds=seeds)

obs_space = len(env.observation_space.high)

agent = get_agent("DQN", obs_space, env.action_space.n, 0.99, lr, seed)

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
os.makedirs(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds"), exist_ok=True)
with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", f"er_buffer.pkl"), "wb") as f:
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

    os.makedirs(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds"), exist_ok=True)
    with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", f"test_{int(eps*100)}.pkl"), "wb") as f:
        pickle.dump(buffer, f)
"""


names = ["er_buffer", "test_0", "test_1", "test_5", "test_10", "test_25", "test_50", "test_75", "test_100"]
rewards, entropies, randomness, unique_states, unique_state_actions = [], [], [], [], []
hll_states, hll_state_actions = [], []

# load training data
with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", "er_buffer" + ".pkl"), "rb") as f:
    buffer = pickle.load(f)
state_limits = []
for axis in range(len(buffer.state[0])):
    state_limits.append(np.min(buffer.state[:,axis]))
    state_limits.append(np.max(buffer.state[:,axis]))
action_limits = copy.deepcopy(state_limits)
action_limits.append(np.min(buffer.action))
action_limits.append(np.max(buffer.action))

for n, name in enumerate(names):
    with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", name + ".pkl"), "rb") as f:
        buffer = pickle.load(f)

    evaluator = Evaluator(test_env, name, buffer.state, buffer.action, buffer.reward, np.invert(buffer.not_done))

    evaluator.train_behavior_policy(epochs=embedding_epochs)

    rewards.append(np.mean(evaluator.get_returns()))
    entropy = np.mean(evaluator.get_bc_entropy())
    entropies.append(entropy)

    if test_env == "MiniGrid-LavaGapS7-v0":
        hll_states.append(evaluator.get_unique_states())
        hll_state_actions.append(evaluator.get_unique_state_actions())
        unique_states.append(evaluator.get_unique_states_exact())
        unique_state_actions.append(evaluator.get_unique_state_actions_exact())
    else:
        hll_states.append(evaluator.get_unique_states(limits=state_limits))
        hll_state_actions.append(evaluator.get_unique_state_actions(limits=action_limits))
        unique_states.append(hll_states[-1])
        unique_state_actions.append(hll_state_actions[-1])

    print(name, rewards[n], entropies[n], "/", unique_states[n], hll_states[n], "/",
          unique_state_actions[n], hll_state_actions[n])


#########################
#       Plotting        #
#########################
os.makedirs(os.path.join("results", "correlation", test_env), exist_ok=True)

### Entropy

plt.figure(figsize=(4, 3))
plt.hlines(y=entropies[0], xmin=0, xmax=1, color="C1", label="ER buffer")
plt.plot(probas, entropies[1:], "-o", color="C0", label="$\epsilon$-greedy")
plt.plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), linestyle="dotted", color="black")
plt.xlabel("$\epsilon$")
plt.ylabel("Entropy")
plt.legend(loc="lower right", fontsize="x-small")
plt.title("-".join(test_env.split("-")[:-1]))
plt.tight_layout()
plt.savefig(os.path.join("results", "correlation", test_env, f"entropy_{len(seeds)}_seeds.pdf"))
plt.close()

### State / State-Action approximation

if test_env == "MiniGrid-LavaGapS7-v0":

    f, axs = plt.subplots(1, 1, figsize=(4, 3))

    """
    xmin = max(np.min(unique_states), np.min(hll_states))
    xmax = min(np.max(unique_states), np.max(hll_states))
    ymin = xmin
    ymax = xmax
    axs[0].plot(unique_states[1:], hll_states[1:], "-o", color="C1", label="HLL")
    axs[0].plot(np.arange(xmin, xmax, (xmax - xmin) / 100)[:100], np.arange(ymin, ymax, (ymax - ymin) / 100)[:100],
                linestyle="dotted", color="black")
    axs[0].set_xlabel("Unique States (exact)")
    axs[0].set_ylabel("Unique States estimate")
    axs[0].legend(loc="lower right", fontsize="x-small")
    axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    """
    xmin = max(np.min(unique_state_actions), np.min(hll_state_actions))
    xmax = min(np.max(unique_state_actions), np.max(hll_state_actions))
    ymin = xmin
    ymax = xmax
    axs.plot(unique_state_actions[1:], hll_state_actions[1:], "-o", color="C1", label="$\epsilon$-greedy")
    axs.plot(np.arange(xmin, xmax, (xmax - xmin) / 100)[:100], np.arange(ymin, ymax, (ymax - ymin) / 100)[:100],
                linestyle="dotted", color="black")
    axs.set_xlabel("Unique State-Actions (exact)")
    axs.set_ylabel("Unique State-Actions estimate")
    axs.legend(loc="lower right", fontsize="x-small")
    #axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.title("1 seed" if len(seeds) == 1 else "no seeds" if len(seeds) == 0 else f"{len(seeds)} seeds")
    f.tight_layout()
    plt.savefig(os.path.join("results", "correlation", test_env, f"approx_quality_{len(seeds)}_seeds.pdf"))
    plt.close()

### epsilon dependence on State and State-Action

f, axs = plt.subplots(1, 1, figsize=(4, 3))

"""
axs[0].plot(probas, hll_states[1:], "-o", color="C2", label="HLL")
axs[0].axhline(y=hll_states[0], color="C3", label="er_buffer")
axs[0].set_xlabel("$\epsilon$")
axs[0].set_ylabel("Unique States estimate")
axs[0].legend(loc="lower right", fontsize="x-small")
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
"""

axs.plot(probas, hll_state_actions[1:], "-o", color="C2", label="$\epsilon$-greedy")
axs.axhline(y=hll_state_actions[0], color="C1", label="ER buffer")
axs.set_xlabel("$\epsilon$")
axs.set_ylabel("Unique state-action pairs")
axs.legend(loc="upper left", fontsize="x-small")
axs.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.autoscale()
plt.title("-".join(test_env.split("-")[:-1]))
f.tight_layout()
plt.savefig(os.path.join("results", "correlation", test_env, f"state_eps_dep_{len(seeds)}_seeds.pdf"))
plt.close()

### Projections

if test_env == "MiniGrid-LavaGapS7-v0":
    rng = np.random.default_rng(seed)
    random_encoder = rng.standard_normal((buffer.state.shape[1], 2))
else:
    random_encoder = np.eye(2)

fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)
axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]

for n, name in enumerate(names):
    with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", name + ".pkl"), "rb") as f:
        buffer = pickle.load(f)

    buffer.state = buffer.state @ random_encoder

    s = None if test_env == "MiniGrid-LavaGapS7-v0" else 0.5
    axs[n].scatter(buffer.state[:-1, 0], buffer.state[:-1, 1], c=[f"C{int(a)}" for a in buffer.action[:-1, 0]], s = s)

    if n == 0:
        axs[n].set_title("ER buffer")
    else:
        axs[n].set_title(f"$\epsilon$ = {float(name.split('_')[1]) / 100}")

if test_env == "MiniGrid-LavaGapS7-v0":
    fig.text(0.54, 0.01, 'dim1', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'dim2', va='center', rotation='vertical', fontsize=14)
else:
    fig.text(0.54, 0.01, 'position (m)', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'velocity (m/s)', va='center', rotation='vertical', fontsize=14)

fig.text(0.52, 0.97, "-".join(test_env.split("-")[:-1]), fontsize=16, ha="center")
fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
plt.savefig(os.path.join("results", "correlation", test_env, f"projections_{len(seeds)}_seeds.png"))
plt.close()