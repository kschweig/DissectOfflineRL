import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .networks import BC, Embedding, AutoEncoder
from .training import update, evaluate
from .datasets import BCSet, StateActionEmbeddingSet, AESet
from .utils import entropy, BColors
from hyperloglog import HyperLogLog
import matplotlib.pyplot as plt
from math import sqrt
import copy
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
                 workers=0,
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

        # Dataset, last state and actions are meaningless
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        # auxiliary parameters
        self.workers = workers
        self.seed = seed

        # could be that dataset contains not every action, then one can pass the correct number of actions
        self.num_actions = num_actions if num_actions is not None else np.max(self.actions) + 1

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # behavioral cloning network
        self.behavioral_trained = False
        self.behavioral = BC(num_state=self.states.shape[1], num_actions=self.num_actions, seed=self.seed).to(device)

        # state embedding network
        self.state_embedding_trained = False
        self.state_embedding = Embedding(num_state=self.states.shape[1], num_embedding=2, seed=self.seed).to(device)

        # state-action embedding network
        self.state_action_embedding_trained = False
        self.state_action_embedding = Embedding(num_state=self.states.shape[1], num_embedding=2, seed=self.seed).to(device)

        # state ae network
        self.state_ae_trained = False
        self.state_ae = AutoEncoder(num_state=self.states.shape[1], num_embedding=2, seed=self.seed).to(device)

        # state-action ae network
        self.state_action_ae_trained = False
        self.state_action_ae = AutoEncoder(num_state=self.states.shape[1], num_embedding=2, seed=self.seed).to(device)

        # copies that stay random
        self.random_state_embedding = copy.deepcopy(self.state_embedding)
        self.random_state_action_embedding = copy.deepcopy(self.state_action_embedding)

        # limits for estimation
        self.limits = [None] * 8

    def evaluate(self, state_limits=None, action_limits=None,
                 epochs=10, batch_size=64, lr=1e-3,
                 subsample=1., verbose=False):

        assert 0 <= subsample <= 1, f"subsample must be in [0;1] but is {subsample}."

        self.train_behavior_policy(epochs, batch_size, lr, verbose)

        rewards = self.get_rewards()
        sparsity = self.get_sparsity()
        ep_lengths = self.get_episode_lengths()
        entropies = self.get_bc_entropy()

        unique_states = self.get_unique_states(limits=state_limits)
        unique_state_actions = self.get_unique_state_actions(limits=action_limits)

        """
        plot_histograms(output, normalized_reward, ep_lengths, share_unique_states, entropies, self.actions,
                        sparsity)
        """

        print("-"*50)
        print("Min / Mean / Max Return: \t\t", f"{round(np.min(rewards), 2)} / {round(np.mean(rewards), 2)} "
                                             f"/ {round(np.max(rewards), 2)}")
        print("Unique States: \t", f"{unique_states}")
        print("Unique State-Actions: \t", f"{unique_state_actions}")
        print("Min / Mean / Max Entropy: \t", f"{round(np.min(entropies), 2)} / {round(np.mean(entropies), 2)} "
                                              f"/ {round(np.max(entropies), 2)}")
        print("Min / Mean / Max Sparsity: \t", f"{round(np.min(sparsity), 2)} / "
                                               f"{round(np.mean(sparsity), 2)} "
                                               f"/ {round(np.max(sparsity), 2)}")
        print("Min / Mean / Max Episode Length: \t", f"{round(np.min(ep_lengths), 2)} / "
                                                     f"{round(np.mean(ep_lengths), 2)} "
                                                     f"/ {round(np.max(ep_lengths), 2)}")
        print("-" * 50)

        return [self.environment, self.buffer_type,
                (np.mean(rewards), np.std(rewards)),
                unique_states,
                unique_state_actions,
                (np.mean(entropies), np.std(entropies)),
                (np.mean(sparsity), np.std(sparsity)),
                (np.mean(ep_lengths), np.std(ep_lengths))
                ]

    def get_rewards(self):

        rewards, ep_reward = list(), 0

        for i, done in enumerate(self.dones):
            ep_reward += self.rewards[i].item()
            if done:
                rewards.append(ep_reward)
                ep_reward = 0

        return rewards

    @staticmethod
    def get_normalized_rewards(rewards, random_reward, optimal_reward):
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
            x = x.to(next(self.behavioral.parameters()).device)
            entropies.extend(entropy(self.behavioral(x)))

        # calculate entropy
        entropies = np.asarray(entropies)

        return entropies

    def get_similarity_distance(self):
        states = torch.FloatTensor(self.states)[:len(self.dones)]
        with torch.no_grad():
            states = states.to(next(self.behavioral.parameters()).device)
            states = self.state_embedding.embed(states).cpu().numpy()

        rng = np.random.default_rng(self.seed)

        ep_distances = []
        general_distances = []

        dones = []
        for d, done in enumerate(self.dones):
            if done:
                dones.append(d + 1)

        start = 0
        for end in dones:
            ep_states = states[start:end]
            for s, state in enumerate(ep_states):
                idx = rng.integers(len(ep_states))
                # in case the same state is sampled by chance
                while idx == s or np.allclose(state, ep_states[idx]):
                    idx = rng.integers(len(ep_states))
                    if len(ep_states) == 1:
                        break
                if np.allclose(state, ep_states[idx]):
                    continue
                distance = (state - ep_states[idx]).reshape(-1,)
                ep_distances.append(np.linalg.norm(distance))
            start = end

        for s, state in enumerate(states):
            idx = rng.integers(len(states))
            # in case the same state is sampled by chance
            while idx == s:
                idx = rng.integers(len(states))
            distance = (state - states[idx]).reshape(-1, )
            general_distances.append(np.linalg.norm(distance))

        return np.mean(general_distances) / np.mean(ep_distances)

    def get_state_pseudo_coverage(self, no_cells=100, use_random=False):
        states = torch.FloatTensor(self.states)[:len(self.dones)]
        with torch.no_grad():
            if use_random:
                states = states.to(next(self.random_state_embedding.parameters()).device)
                states = self.random_state_embedding.embed(states).cpu().numpy()
                if self.limits[4] is None:
                    self.limits[4] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                      np.min(states[:, 1]), np.max(states[:, 1]))
                limits = self.limits[4]
            else:
                states = states.to(next(self.state_embedding.parameters()).device)
                states = self.state_embedding.embed(states).cpu().numpy()
                if self.limits[0] is None:
                    self.limits[0] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                      np.min(states[:, 1]), np.max(states[:, 1]))
                limits = self.limits[0]
        return self.calc_coverage(states, limits, no_cells)

    def get_state_action_pseudo_coverage(self, no_cells=100, use_random=False):
        states = torch.FloatTensor(self.states + self.actions)[:len(self.dones)]
        with torch.no_grad():
            if use_random:
                states = states.to(next(self.random_state_action_embedding.parameters()).device)
                states = self.random_state_action_embedding.embed(states).cpu().numpy()
                if self.limits[5] is None:
                    self.limits[5] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                      np.min(states[:, 1]), np.max(states[:, 1]))
                limits = self.limits[5]
            else:
                states = states.to(next(self.state_action_embedding.parameters()).device)
                states = self.state_action_embedding.embed(states).cpu().numpy()
                if self.limits[1] is None:
                    self.limits[1] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                      np.min(states[:, 1]), np.max(states[:, 1]))
                limits = self.limits[1]
        return self.calc_coverage(states, limits, no_cells)

    def get_state_ae_pseudo_coverage(self, no_cells=100):
        states = torch.FloatTensor(self.states)
        with torch.no_grad():
            states = states.to(next(self.state_ae.parameters()).device)[:len(self.dones)]
            states = self.state_ae.embed(states).cpu().numpy()
            if self.limits[2] is None:
                self.limits[2] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                  np.min(states[:, 1]), np.max(states[:, 1]))
        return self.calc_coverage(states, self.limits[2], no_cells)

    def get_state_action_ae_pseudo_coverage(self, no_cells=100):
        states = torch.FloatTensor(self.states + self.actions)[:len(self.dones)]
        with torch.no_grad():
            states = states.to(next(self.state_action_ae.parameters()).device)
            states = self.state_action_ae.embed(states).cpu().numpy()
            if self.limits[3] is None:
                self.limits[3] = (np.min(states[:, 0]), np.max(states[:, 0]),
                                  np.min(states[:, 1]), np.max(states[:, 1]))
        return self.calc_coverage(states, self.limits[3], no_cells)

    def calc_coverage(self, states, limits, no_cells=100):
        return self.get_unique_states(states = states, limits=limits)

    """
    @staticmethod
    def calc_coverage(states, limits, no_cells=100):
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        targets = copy.deepcopy(states)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(targets, axis=1)
        return np.mean(np.sum(states * targets, axis=1))
    """

    """
    @staticmethod
    def calc_coverage(states, limits, no_cells=100):
        grid = np.zeros((no_cells, no_cells))
        xmin, xmax, ymin, ymax = limits
        states[:, 0] -= xmin
        states[:, 1] -= ymin
        states[:, 0] = states[:, 0] / xmax * no_cells
        states[:, 1] = states[:, 1] / ymax * no_cells

        for state in states:
            # adjust for the maximum outer most points!
            x = max(min(int(state[0]), no_cells - 1), 0)
            y = max(min(int(state[1]), no_cells - 1), 0)
            grid[x, y] = 1

        return np.sum(grid) / no_cells ** 2
    """


    def plot_states(self, use_random=False, path=None):
        states = torch.FloatTensor(self.states)[:len(self.dones)]
        with torch.no_grad():
            if use_random:
                states = states.to(next(self.random_state_embedding.parameters()).device)
                states = self.random_state_embedding.embed(states).cpu().numpy()
            else:
                states = states.to(next(self.state_embedding.parameters()).device)
                states = self.state_embedding.embed(states).cpu().numpy()
        self._plot_states(states, path)

    def plot_state_actions(self, use_random=False, path=None):
        states = torch.FloatTensor(self.states + self.actions)[:len(self.dones)]
        with torch.no_grad():
            if use_random:
                states = states.to(next(self.random_state_action_embedding.parameters()).device)
                states = self.random_state_action_embedding.embed(states).cpu().numpy()
            else:
                states = states.to(next(self.state_action_embedding.parameters()).device)
                states = self.state_action_embedding.embed(states).cpu().numpy()
        self._plot_states(states, path)

    def plot_states_ae(self, path=None):
        states = torch.FloatTensor(self.states)[:len(self.dones)]
        with torch.no_grad():
            states = states.to(next(self.state_ae.parameters()).device)
            states = self.state_ae.embed(states).cpu().numpy()
        self._plot_states(states, path)

    def plot_state_actions_ae(self, path=None):
        states = torch.FloatTensor(self.states + self.actions)[:len(self.dones)]
        with torch.no_grad():
            states = states.to(next(self.state_action_ae.parameters()).device)
            states = self.state_action_ae.embed(states).cpu().numpy()
        self._plot_states(states, path)

    def _plot_states(self, states, path):
        plt.figure(figsize=(4,3))
        plt.scatter(states[:, 0], states[:, 1])

        dones = []
        for d, done in enumerate(self.dones):
            if done:
                dones.append(d + 1)

        plt.title(f"{self.environment} @ {self.buffer_type}")
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.plot(states[:dones[0], 0], states[:dones[0], 1], "-o", color="black")
        plt.plot(states[dones[len(dones) // 2]:dones[len(dones) // 2 + 1], 0],
                 states[dones[len(dones) // 2]:dones[len(dones) // 2 + 1], 1], "-o", color="red")
        plt.plot(states[dones[-2]:dones[-1], 0], states[dones[-2]:dones[-1], 1], "-o", color="blue")
        plt.plot(states[0, 0], states[0, 1], "*", color="black", markersize=12)
        plt.plot(states[dones[len(dones) // 2], 0], states[dones[len(dones) // 2], 1], "*", color="red", markersize=12)
        plt.plot(states[dones[-2], 0], states[dones[-2], 1], marker="*", color="blue", markersize=12)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(path))
        plt.close()

    def get_unique_states(self, states=None, limits=None):
        if states is None:
            states = copy.deepcopy(self.states)

        for axis in range(len(states[0])):
            if limits is None:
                axmin, axmax = np.min(states[:, axis]), np.max(states[:, axis])
            else:
                axmin, axmax = limits[axis*2:axis*2+2]

            states[:, axis] = np.digitize(states[:, axis],
                                          np.linspace(axmin, axmax, num=100))
        states.astype(int)

        hll = HyperLogLog(0.01)
        for state in tqdm(states,
                          desc=f"Search for Unique States in whole dataset ({self.environment} @ {self.buffer_type})",
                          total=len(states)):
            hll.add(",".join([str(s) for s in state]))

        return len(hll)

    def get_unique_state_actions(self, states=None, actions=None, limits=None):
        if states is None:
            states = copy.deepcopy(self.states)
        if actions is None:
            actions = copy.deepcopy(self.actions)

        states += actions

        return self.get_unique_states(states, limits)

    def get_unique_states_exact(self):
        unique = []
        for i, done in tqdm(enumerate(self.dones),
                            desc=f"Search exact for Unique States in whole dataset ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            found = False
            for unique_state in unique:
                if np.allclose(self.states[i], unique_state):
                    found = True
                    break
            if not found:
                unique.append(self.states[i])
        return len(unique)

    def get_unique_state_actions_exact(self):
        unique = []
        for i, done in tqdm(enumerate(self.dones),
                            desc=f"Search exact for Unique State-Action pairs in whole dataset ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            found = False
            for unique_state_action in unique:
                if np.allclose(self.states[i] + self.actions[i], unique_state_action):
                    found = True
                    break
            if not found:
                unique.append(self.states[i] + self.actions[i])
        return len(unique)

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

    def train_state_embedding(self, epochs=10, batch_size=64, lr=1e-3, negative_sampling=1, verbose=False):

        # state action embedding set without action given uses just the state
        dl = DataLoader(StateActionEmbeddingSet(states=self.states, dones=self.dones, seed=self.seed,
                                                negative_sampling=negative_sampling),
                        batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)

        optimizer = Adam(self.state_embedding.parameters(), lr=lr)
        loss = nn.BCEWithLogitsLoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.state_embedding, dl, loss))

        for ep in tqdm(range(epochs), desc=f"Training State Embedding ({self.environment} @ {self.buffer_type})"):
            errs = update(self.state_embedding, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep+1}, loss: {np.mean(errs)}")



        self.state_embedding_trained = True

    def train_state_action_embedding(self, epochs=10, batch_size=64, lr=1e-3, negative_sampling=1, verbose=False):

        dl = DataLoader(StateActionEmbeddingSet(states=self.states, dones=self.dones, seed=self.seed,
                                                actions=self.actions,  negative_sampling=negative_sampling),
                        batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)

        optimizer = Adam(self.state_action_embedding.parameters(), lr=lr)
        loss = nn.BCEWithLogitsLoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.state_action_embedding, dl, loss))

        for ep in tqdm(range(epochs), desc=f"Training State-Action Embedding ({self.environment} @ {self.buffer_type})"):
            errs = update(self.state_action_embedding, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep+1}, loss: {np.mean(errs)}")

        self.state_action_embedding_trained = True

    def train_state_ae(self, epochs=10, batch_size=64, lr=1e-3, verbose=False):

        dl = DataLoader(AESet(states=self.states),
                        batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)

        optimizer = Adam(self.state_ae.parameters(), lr=lr)
        loss = nn.MSELoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.state_ae, dl, loss))

        for ep in tqdm(range(epochs),
                       desc=f"Training State AutoEncoder ({self.environment} @ {self.buffer_type})"):
            errs = update(self.state_ae, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep + 1}, loss: {np.mean(errs)}")

        self.state_ae_trained = True

    def train_state_action_ae(self, epochs=10, batch_size=64, lr=1e-3, verbose=False):

        dl = DataLoader(AESet(states=self.states, actions=self.actions),
                        batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)

        optimizer = Adam(self.state_action_ae.parameters(), lr=lr)
        loss = nn.MSELoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.state_action_ae, dl, loss))

        for ep in tqdm(range(epochs),
                       desc=f"Training State-Action AutoEncoder ({self.environment} @ {self.buffer_type})"):
            errs = update(self.state_action_ae, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep + 1}, loss: {np.mean(errs)}")

        self.state_action_ae_trained = True





