import os
import torch
import pickle
import warnings
import numpy as np
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

test_env = "MountainCar-v0"#"MiniGrid-LavaGapS7-v0"#"Breakout-MinAtar-v0"#

# keep training parameters for online training fixed, the experiment does not interfere here.
seed = 42
seed_list = [[seed],
             [seed, seed+100, seed+200],
             [seed, seed+100, seed+200, seed+300, seed+400],
             []]

# no loop to make it runnable by multiple processes
seeds = seed_list[2]

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


names = ["er_buffer", "test_0", "test_1", "test_5", "test_10", "test_25", "test_50", "test_75", "test_100"]
rewards, entropies, randomness, unique_states, unique_state_actions = [], [], [], [], []
scs, sacs, scs_random, sacs_random, scs_pca, sacs_pca, scs_ae, sacs_ae = [], [], [], [], [], [], [], []

# load training data
with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", "er_buffer" + ".pkl"), "rb") as f:
    buffer = pickle.load(f)

# train evaluator
state_evaluator = Evaluator(test_env, "er_buffer", buffer.state, buffer.action, buffer.reward, np.invert(buffer.not_done))
state_evaluator.train_state_embedding(epochs=embedding_epochs)
state_evaluator.get_state_pseudo_coverage()
state_evaluator.train_state_action_embedding(epochs=embedding_epochs)
state_evaluator.get_state_action_pseudo_coverage()
state_evaluator.train_state_ae(epochs=embedding_epochs)
state_evaluator.get_state_ae_pseudo_coverage()
state_evaluator.train_state_action_ae(epochs=embedding_epochs)
state_evaluator.get_state_action_ae_pseudo_coverage()
state_evaluator.get_state_pseudo_coverage(use_random=True)
state_evaluator.get_state_action_pseudo_coverage(use_random=True)

# train PCA and get limits
state_pca = PCA(n_components=2)
state_action_pca = PCA(n_components=2)
pred = state_pca.fit_transform(buffer.state[:buffer_size])
state_pca_limits = (np.min(pred[:, 0]), np.max(pred[:, 0]), np.min(pred[:, 1]), np.max(pred[:, 1]))
pred = state_action_pca.fit_transform(buffer.state[:buffer_size] + buffer.action[:buffer_size])
state_action_pca_limits = (np.min(pred[:, 0]), np.max(pred[:, 0]), np.min(pred[:, 1]), np.max(pred[:, 1]))
del pred

for n, name in enumerate(names):
    with open(os.path.join("data", "ex_corr", test_env, f"{len(seeds)}_seeds", name + ".pkl"), "rb") as f:
        buffer = pickle.load(f)

    evaluator = Evaluator(test_env, name, buffer.state, buffer.action, buffer.reward, np.invert(buffer.not_done))

    evaluator.train_behavior_policy(epochs=5)
    # always use the same embedding
    evaluator.state_embedding = state_evaluator.state_embedding
    evaluator.state_action_embedding = state_evaluator.state_action_embedding
    evaluator.random_state_embedding = state_evaluator.random_state_embedding
    evaluator.random_state_action_embedding = state_evaluator.random_state_action_embedding
    evaluator.state_ae = state_evaluator.state_ae
    evaluator.state_action_ae = state_evaluator.state_action_ae
    evaluator.limits = state_evaluator.limits

    pca_states = state_pca.transform(buffer.state[:buffer_size])
    pca_state_actions = state_pca.transform((buffer.state + buffer.action)[:buffer_size])

    # plot embedded dimensions
    os.makedirs(os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds"), exist_ok=True)
    evaluator.plot_states(path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"nsp_{name}.png"))
    evaluator.plot_state_actions(path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"nsap_{name}.png"))
    evaluator.plot_states(use_random=True, path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"nsp_r_{name}.png"))
    evaluator.plot_state_actions(use_random=True, path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"nsap_r_{name}.png"))
    evaluator.plot_states_ae(path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"sae_{name}.png"))
    evaluator.plot_state_actions_ae(path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"saae_{name}.png"))
    #evaluator._plot_states(pca_states, path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"spca_{name}.png"))
    #evaluator._plot_states(pca_state_actions, path=os.path.join("results", "img", "projections", test_env, f"{len(seeds)}seeds", f"sapca_{name}.png"))

    rewards.append(np.mean(evaluator.get_rewards()))
    entropy = np.mean(evaluator.get_bc_entropy())
    entropies.append(entropy)

    unique_states.append(evaluator.get_unique_states())
    unique_state_actions.append(evaluator.get_unique_state_actions())

    scs.append(evaluator.get_state_pseudo_coverage())
    sacs.append(evaluator.get_state_action_pseudo_coverage())

    scs_random.append(evaluator.get_state_pseudo_coverage(use_random=True))
    sacs_random.append(evaluator.get_state_action_pseudo_coverage(use_random=True))

    scs_ae.append(evaluator.get_state_ae_pseudo_coverage())
    sacs_ae.append(evaluator.get_state_action_ae_pseudo_coverage())

    scs_pca.append(evaluator.calc_coverage(pca_states, state_pca_limits))
    sacs_pca.append(evaluator.calc_coverage(pca_state_actions, state_action_pca_limits))

    # print("exact unique states: ", evaluator.get_unique_states_exact())
    # print("exact unique state-action pairs: ", evaluator.get_unique_state_actions_exact())

    print(name, rewards[n], entropies[n], "/", unique_states[n], scs[n], scs_random[n], scs_ae[n], scs_pca[n], "/",
          unique_state_actions[n], sacs[n], sacs_random[n], sacs_ae[n], sacs_pca[n])


#########################
#       Plotting        #
#########################
os.makedirs(os.path.join("results", "img", "correlation", test_env), exist_ok=True)
f, axs = plt.subplots(1, 3, figsize=(12, 3))

axs[0].plot(probas, entropies[1:], "-o", color="C0", label="BC")
axs[0].plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), linestyle="dotted", color="black")
axs[0].set_xlabel("$\epsilon$")
axs[0].set_ylabel("Entropy")
axs[0].legend(loc="upper left", fontsize="x-small")

xmin = max(np.min(unique_states),
           min(np.min(scs), np.min(scs_random), np.min(scs_ae), np.min(scs_pca)))
xmax = min(np.max(unique_states),
           max(np.max(scs), np.max(scs_random), np.max(scs_ae), np.max(scs_pca)))
ymin = max(np.min(unique_states), min(np.min(scs), np.min(scs_random), np.min(scs_ae), np.min(scs_pca)))
ymax = min(np.max(unique_states), max(np.max(scs), np.max(scs_random), np.max(scs_ae), np.max(scs_pca)))
axs[1].plot(unique_states[1:], scs[1:], "-o", color="C1", label="NSP")
axs[1].plot(unique_states[1:], scs_random[1:], "-o", color="C2", label="RE")
axs[1].plot(unique_states[1:], scs_pca[1:], "-o", color="C3", label="PCA")
axs[1].plot(unique_states[1:], scs_ae[1:], "-o", color="C4", label="AE")
axs[1].plot(np.arange(xmin, xmax, (xmax - xmin) / 100)[:100], np.arange(ymin, ymax, (ymax - ymin) / 100)[:100],
            linestyle="dotted", color="black")
axs[1].set_xlabel("Unique States")
axs[1].set_ylabel("Pseudo-State-Coverage")
axs[1].legend(loc="upper left", fontsize="x-small")
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

xmin = max(np.min(unique_state_actions),
           min(np.min(sacs), np.min(sacs_random), np.min(sacs_ae), np.min(sacs_pca)))
xmax = min(np.max(unique_state_actions),
           max(np.max(sacs), np.max(sacs_random), np.max(sacs_ae), np.max(sacs_pca)))
ymin = max(np.min(unique_state_actions),
           min(np.min(sacs), np.min(sacs_random), np.min(sacs_ae), np.min(sacs_pca)))
ymax = min(np.max(unique_state_actions),
           max(np.max(sacs), np.max(sacs_random), np.max(sacs_ae), np.max(sacs_pca)))
axs[2].plot(unique_state_actions[1:], sacs[1:], "-o", color="C1", label="NSAP")
axs[2].plot(unique_state_actions[1:], sacs_random[1:], "-o", color="C2", label="RE")
axs[2].plot(unique_state_actions[1:], sacs_pca[1:], "-o", color="C3", label="PCA")
axs[2].plot(unique_state_actions[1:], sacs_ae[1:], "-o", color="C4", label="AE")
axs[2].plot(np.arange(xmin, xmax, (xmax - xmin) / 100)[:100], np.arange(ymin, ymax, (ymax - ymin) / 100)[:100],
            linestyle="dotted", color="black")
axs[2].set_xlabel("Unique State-Actions")
axs[2].set_ylabel("Pseudo-State-Action-Coverage")
axs[2].legend(loc="upper left", fontsize="x-small")
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

f.text(0.5, 0.92, "1 seed" if len(seeds) == 1 else "no seeds" if len(seeds) == 0 else f"{len(seeds)} seeds",
       ha='center', fontsize="large")
f.tight_layout()
plt.savefig(os.path.join("results", "img", "correlation", test_env, f"{len(seeds)}_seeds.pdf"))
