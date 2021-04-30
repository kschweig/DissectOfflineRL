import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .networks import BC
from .training import update, evaluate
from .datasets import BCSet
from .utils import entropy, BColors
from .plotting import plot_histograms
from .latex import create_latex_table



class Evaluator():

    def __init__(self,
                 environment: str,
                 buffer_type: str,
                 states:np.ndarray,
                 actions:np.ndarray,
                 rewards:np.ndarray,
                 dones:np.ndarray,
                 workers=4,
                 seed=42,
                 num_actions=None):

        assert len(states.shape) == 2, f"States must be of dimension (ds_size, feature_size), were ({states.shape})"
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        assert len(actions.shape) == 2, f"Actions must be of dimension (ds_size, 1), were ({actions.shape})"
        if len(rewards.shape) == 1:
            rewards = rewards.reshape(-1, 1)
        assert len(rewards.shape) == 2, f"Rewards must be of dimension (ds_size, 1), were ({actions.shape})"
        if len(dones.shape) == 1:
            dones = dones.reshape(-1, 1)
        assert len(dones.shape) == 2, f"Dones must be of dimension (ds_size, 1), were ({actions.shape})"

        # task information
        self.environment = environment
        self.buffer_type = buffer_type

        # Dataset
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        # auxiliary parameters
        self.workers = workers
        self.seed = seed

        # could be that dataset contains not every action, then one can pass the correct number of actions
        self.num_actions = num_actions if num_actions is not None else np.max(self.actions) + 1

        # behavioral cloning network
        self.behavioral_trained = False
        self.behavioral = BC(num_state=self.states.shape[1], num_actions=self.num_actions, seed=self.seed)

    def evaluate(self, output=os.path.join("results", "ds_eval","test"), random_reward = 0, optimal_reward = 1,
                 epochs=10, batch_size=64, lr=1e-3,
                 subsample=1., verbose=False):

        assert 0 <= subsample <= 1, f"subsample must be in [0;1] but is {subsample}."

        self.train_behavior_policy(epochs, batch_size, lr, verbose)

        rewards = self.get_rewards()
        sparsity = self.get_sparsity()
        ep_lengths = self.get_episode_lengths()
        entropies = self.get_bc_entropy()

        # normalize states
        self.states /= np.linalg.norm(self.states, axis=1, keepdims=True)

        unique_states_episode = self.get_unique_states_episode()
        unique_states = self.get_unique_states()

        plot_histograms(output, rewards, ep_lengths, unique_states_episode, entropies, self.actions,
                        sparsity)

        normalized_reward = self.get_normalized_rewards(rewards, random_reward, optimal_reward)

        print("-"*50)
        print("Min / Mean / Max Reward: \t\t", f"{round(np.min(rewards), 2)} / {round(np.mean(rewards), 2)} "
                                             f"/ {round(np.max(rewards), 2)}")
        print("Min / Mean / Max Normalized Reward: \t\t", f"{round(np.min(normalized_reward), 2)} / "
                                               f"{round(np.mean(normalized_reward), 2)} "
                                               f"/ {round(np.max(normalized_reward), 2)}")
        print("Min / Mean / Max Entropy: \t", f"{round(np.min(entropies), 2)} / {round(np.mean(entropies), 2)} "
                                              f"/ {round(np.max(entropies), 2)}")
        print("Min / Mean / Max Episode Length: \t", f"{round(np.min(ep_lengths), 2)} / "
                                                   f"{round(np.mean(ep_lengths), 2)} "
                                                   f"/ {round(np.max(ep_lengths), 2)}")
        print("Min / Mean / Max Sparsity: \t", f"{round(np.min(sparsity), 2)} / "
                                               f"{round(np.mean(sparsity), 2)} "
                                               f"/ {round(np.max(sparsity), 2)}")
        print("Min / Mean / Max Unique States per Episode: \t", f"{round(np.min(unique_states_episode), 2)} / "
                                               f"{round(np.mean(unique_states_episode), 2)} "
                                               f"/ {round(np.max(unique_states_episode), 2)}")
        print("Share of unique states in dataset: \t", f"{round(unique_states, 5)}")
        print("-" * 50)

        return [self.environment, self.buffer_type,
                (np.mean(rewards), np.std(rewards)), (np.mean(normalized_reward), np.std(normalized_reward)),
                (np.mean(entropies), np.std(entropies)),
                (np.mean(ep_lengths), np.std(ep_lengths)), (np.mean(sparsity), np.std(sparsity)),
                (np.mean(unique_states_episode), np.std(unique_states_episode)), unique_states]

    def get_rewards(self):

        rewards, ep_reward = list(), 0

        for i, done in enumerate(self.dones):
            ep_reward += self.rewards[i].item()
            if done:
                rewards.append(ep_reward)
                ep_reward = 0

        return rewards

    def get_normalized_rewards(self, rewards, random_reward, optimal_reward):
        normalized_reward = []
        for reward in rewards:
            normalized_reward.append((reward - random_reward) / (optimal_reward - random_reward))
        return normalized_reward

    def get_sparsity(self):

        sparsity, num_not_obtained = list(), list()

        for i, done in enumerate(self.dones):
            num_not_obtained.append(self.rewards[i].item() == 0)
            if done:
                sparsity.append(np.mean(num_not_obtained))
                num_not_obtained = list()

        return sparsity

    def get_episode_lengths(self):

        lengths, ep_length = list(), 0

        for i, done in enumerate(self.dones):
            ep_length += 1
            if done:
                lengths.append(ep_length)
                ep_length = 0

        return lengths

    def get_bc_entropy(self):
        if not self.behavioral_trained:
            print(BColors.WARNING + "Attention, behavioral policy was not trained before calling get_bc_entropy!" + BColors.ENDC)

        entropies = []
        dl = DataLoader(BCSet(states=self.states, actions=self.actions), batch_size=512, drop_last=False,
                        shuffle=False, num_workers=self.workers)

        for x, _ in dl:
            entropies.extend(entropy(self.behavioral(x)))

        # calculate maximum entropy and normalize
        max_entropy = entropy(torch.ones((1, self.num_actions)) / self.num_actions)
        entropies = np.asarray(entropies) / max_entropy

        return entropies

    def get_unique_states_episode(self, threshold=0.999):
        unique_states_episode, unique = [], []
        for i, done in tqdm(enumerate(self.dones),
                            desc=f"Search for Unique States per Episode ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            if done:
                unique_states_episode.append(len(unique))
                unique = []

            found = False
            for unique_state in unique:
                if np.dot(self.states[i], unique_state) > threshold:
                    found = True
                    break
            if not found:
                unique.append(self.states[i])

        return unique_states_episode

    def get_unique_states(self, threshold=0.999):
        unique = []
        for i, done in tqdm(enumerate(self.dones),
                            desc=f"Search for Unique States in whole dataset ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            found = False
            for unique_state in unique:
                if np.dot(self.states[i], unique_state) > threshold:
                    found = True
                    break
            if not found:
                unique.append(self.states[i])

        return len(unique) / len(self.states)

    def train_behavior_policy(self, epochs=10, batch_size=64, lr=1e-3, verbose=False):

        dl = DataLoader(BCSet(states=self.states, actions=self.actions), batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)
        optimizer = Adam(self.behavioral.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.behavioral, dl, loss))

        for ep in tqdm(range(epochs), desc=f"Training Behavioral Policy ({self.environment} @ {self.buffer_type})"):
            errs = update(self.behavioral, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep+1}, loss: {np.mean(errs)}")

        self.behavioral_trained = True






