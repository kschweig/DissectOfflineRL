import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .networks import BC, SC, NSC
from .training import update, evaluate
from .datasets import BCSet, VCSet, SCSet, StateSet
from .utils import entropy, BColors



class Evaluator():

    def __init__(self,
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

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        self.workers = workers
        self.seed = seed

        # could be that dataset contains not every action, then one can pass the correct number of actions
        self.num_actions = num_actions if num_actions is not None else np.max(self.actions) + 1

        # behavioral cloning network
        self.behavioral_trained = False
        self.behavioral = BC(num_state=self.states.shape[1], num_actions=self.num_actions, seed=self.seed)

        # behavioral cloning network
        self.value_critic_trained = False
        self.value_critic = BC(num_state=self.states.shape[1], num_actions=self.num_actions, seed=self.seed)

        # Network to assess whether something is the same state.
        self.state_comparator_trained = False
        self.state_comparator = SC(num_state=self.states.shape[1], seed=self.seed)

    def get_rewards(self):

        rewards, ep_reward = list(), 0

        for i, done in enumerate(self.dones):
            ep_reward += self.rewards[i]
            if done:
                rewards.append(ep_reward)
                ep_reward = 0

        return np.min(rewards), np.mean(rewards), np.max(rewards)

    def get_episode_lengths(self):

        lengths, ep_length = list(), 0

        for i, done in enumerate(self.dones):
            ep_length += 1
            if done:
                lengths.append(ep_length)
                ep_length = 0

        return np.min(lengths), np.mean(lengths), np.max(lengths)

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

        return np.min(entropies), np.mean(entropies), np.max(entropies)

    def get_value_estimate(self):
        if not self.behavioral_trained:
            print(BColors.WARNING + "Attention, behavioral policy was not trained before calling get_value_estimate!" + BColors.ENDC)
        if not self.value_critic_trained:
            print(BColors.WARNING + "Attention, value critic was not trained before calling get_value_estimate!" + BColors.ENDC)

        values = []
        dl = DataLoader(VCSet(states=self.states, actions=self.actions, rewards=self.rewards, dones=self.dones),
                        batch_size=512, drop_last=False, shuffle=False, num_workers=self.workers)

        with torch.no_grad():
            for state, _, action, _, _, _ in dl:
                values.extend(self.value_critic(state).gather(1, action).cpu().numpy())

        # de-normalize values
        values += np.max(self.rewards)

        return np.min(values), np.mean(values), np.max(values)

    def get_start_randomness(self, compare_with=1):
        if not self.state_comparator_trained:
            print(BColors.WARNING + "Attention, state comparator was not trained before calling get_start_randomness!" + BColors.ENDC)

        comparison = []
        dl = DataLoader(StateSet(states=self.states, dones=self.dones, starts=True, compare_with=compare_with),
                        batch_size=512, drop_last=False, shuffle=False, num_workers=self.workers)

        with torch.no_grad():
            for state in tqdm(dl):
                comparison.extend(self.state_comparator(state).cpu().numpy())

        return np.min(comparison), np.mean(comparison), np.max(comparison)

    def get_state_randomness(self, compare_with=1):
        if not self.state_comparator_trained:
            print(BColors.WARNING + "Attention, state comparator was not trained before calling get_state_randomness!" + BColors.ENDC)

        comparison = []
        dl = DataLoader(StateSet(states=self.states, dones=self.dones, starts=False, compare_with=compare_with),
                        batch_size=512, drop_last=False, shuffle=False, num_workers=self.workers)

        with torch.no_grad():
            for state in tqdm(dl, desc="estimate state randomness"):
                comparison.extend(self.state_comparator(state).cpu().numpy())

        return np.min(comparison), np.mean(comparison), np.max(comparison)

    def get_episode_intersections(self, treshold=0.98, max_batch_size=512):
        if not self.state_comparator_trained:
            print(BColors.WARNING + "Attention, state comparator was not trained before calling get_episode_intersections!" + BColors.ENDC)

        intersections, inter, first_idx = list(), 0, 0

        for i, done in tqdm(enumerate(self.dones), desc="Calculate Episode Intersections", total=len(self.dones)):
            if i == first_idx:
                continue

            # compare current state to every state in the episode before
            current_state = self.states[i].reshape(1,-1)
            compare_states = self.states[first_idx:i]
            compare_states = np.concatenate((compare_states, np.repeat(current_state, i-first_idx, axis=0)), axis=1)
            compare_states = torch.FloatTensor(compare_states)
            pred = self.state_comparator(compare_states).detach().cpu().numpy()

            # Increment iterations if similarity to at least one prior state is greater than 1.
            # Not using the total number of states as that would double-count those intersections.
            inter += len(np.where(pred > treshold)[0]) > 0

            if done:
                intersections.append(inter)
                inter = 0
                first_idx = i

        return np.min(intersections), np.mean(intersections), np.max(intersections)

    def get_state_determinism(self):
        if not self.state_comparator_trained:
            print(BColors.WARNING + "Attention, next state comparator was not trained before calling get_state_compare!" + BColors.ENDC)
        pass

    def test_state_compare(self, negative_samples=0, sparse_state=False):
        if not self.state_comparator_trained:
            print(BColors.WARNING + "Attention, state comparator was not trained before calling test_state_compare!" + BColors.ENDC)

        comparison = []
        dl = DataLoader(SCSet(states=self.states, negative_samples=negative_samples, sparse_state=sparse_state),
                        batch_size=512, drop_last=False, shuffle=False, num_workers=self.workers)

        with torch.no_grad():
            for state, _ in dl:
                comparison.extend(self.state_comparator(state).cpu().numpy())

        return comparison

    def train_behavior_policy(self, epochs=10, batch_size=64, lr=1e-3, verbose=False):

        dl = DataLoader(BCSet(states=self.states, actions=self.actions), batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)
        optimizer = Adam(self.behavioral.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.behavioral, dl, loss))

        for ep in tqdm(range(epochs), desc="Training Behavioral Policy"):
            errs = update(self.behavioral, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep+1}, loss: {np.mean(errs)}")

        self.behavioral_trained = True

    def train_value_critic(self, epochs=10, batch_size=64, lr=1e-3, horizon=100, verbose=False):
        if not self.behavioral_trained:
            print(BColors.WARNING + "Attention, behavioral policy was not trained before calling train_value_critic!" + BColors.ENDC)

        dl = DataLoader(VCSet(states=self.states, actions=self.actions, rewards=self.rewards, dones=self.dones),
                        batch_size=batch_size, drop_last=True, shuffle=True, num_workers=self.workers)
        optimizer = Adam(self.value_critic.parameters(), lr=lr)
        loss = nn.MSELoss()

        for ep in tqdm(range(epochs), desc="Training Behavioral Critic"):
            losses = []

            for state, next_state, action, next_action, reward, not_done in dl:
                # Compute the target Q value by using the observed data
                with torch.no_grad():
                    target_Q = reward + not_done * np.exp(-1/horizon) * self.value_critic.forward(next_state).gather(1, next_action)

                # Get current Q estimate
                current_Q = self.value_critic.forward(state).gather(1, action)

                # Compute Q loss (Huber loss)
                Q_loss = loss(current_Q, target_Q)

                # Optimize the Q
                optimizer.zero_grad()
                Q_loss.backward()
                optimizer.step()

                losses.append(Q_loss.item())

            if verbose:
                print("Episode", ep+1, "TD-error:", np.mean(losses))

        self.value_critic_trained = True

    def train_state_comparator(self, epochs=10, batch_size=64, lr=1e-3, negative_samples=10, sparse_state=False, verbose=False):

        dl = DataLoader(SCSet(states=self.states, negative_samples=negative_samples, sparse_state=sparse_state),
                        batch_size=batch_size, drop_last=True, shuffle=True, num_workers=self.workers)
        optimizer = Adam(self.state_comparator.parameters(), lr=lr)
        loss = nn.BCELoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.state_comparator, dl, loss))

        for ep in tqdm(range(epochs), desc="Training State Comparator"):
            errs = update(self.state_comparator, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep + 1}, loss: {np.mean(errs)}")

        self.state_comparator_trained = True



