import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import cosine_similarity


class BCSet(Dataset):

    def __init__(self, states, actions):
        super(BCSet, self).__init__()

        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return torch.FloatTensor(self.states[item]), torch.LongTensor(self.actions[item])


class VCSet(Dataset):

    def __init__(self, states, actions, rewards, dones):
        super(VCSet, self).__init__()

        self.states = states
        self.actions = actions
        self.rewards = np.zeros_like(rewards)

        # calculate total reward per episode
        self.total_rewards = []
        reward = 0
        for i in range(len(rewards)):
            reward += rewards[i]
            if dones[i] or i == len(rewards) - 1:
                self.total_rewards.append(reward)
                reward = 0

        # calculate remaining reward until end of episode
        idx = 0
        reward = self.total_rewards[idx]
        for i in range(len(rewards)):
            self.rewards[i] = reward
            reward -= rewards[i]
            if dones[i]:
                idx += 1
                reward = self.total_rewards[idx]

        self.not_dones = np.invert(dones)

    def __len__(self):
        return len(self.states) - 1

    def __getitem__(self, item):
        return torch.FloatTensor(self.states[item]), torch.LongTensor(self.actions[item]), \
               torch.FloatTensor(self.rewards[item]), torch.FloatTensor(self.not_dones[item])


class SCSet(Dataset):

    def __init__(self, states, negative_samples=10, sparse_state=False, treshold=0.95):
        super(SCSet, self).__init__()

        assert treshold >=-1 and treshold <= 1, f"treshold parameter must be in [-1,1] but is {treshold}."

        self.states = states
        self.negative_samples = 1 - (1 / (negative_samples + 1)) if negative_samples >= 0 else -1
        self.sparse_state = sparse_state
        self.treshold = treshold
        self.rng = np.random.default_rng(seed=42)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):

        if self.rng.random() < self.negative_samples or self.negative_samples < 0:
            if self.sparse_state:
                equal = True
                while equal:
                    idx = self.rng.integers(self.__len__())
                    if idx == item:
                        continue
                    equal = cosine_similarity(self.states[item], self.states[idx]) > self.treshold
            else:
                idx = item
                while idx == item:
                    idx = self.rng.integers(self.__len__())

            return np.concatenate((self.states[item], self.states[idx])), np.zeros((1), dtype=np.float32)

        return np.concatenate((self.states[item], self.states[item])), np.ones((1), dtype=np.float32)


class StateSet(Dataset):

    def __init__(self, states, dones, starts=True, compare_with=1):
        super(StateSet, self).__init__()

        assert compare_with >= 1 and isinstance(compare_with, int), f"compare_with must be positive integer, is {compare_with} type={type(compare_with)}"

        # dones are used to indicate starting state, therefore shifted!
        self.dones = np.ones_like(dones)
        self.dones[1:] = dones[:-1]
        self.rng = np.random.default_rng()
        self.compare_with = compare_with

        if starts:
            self.states = states[np.where(dones == 1)[0]]
        else:
            self.states = states

        print(self.__len__())

    def __len__(self):
        return len(self.states) * self.compare_with

    def __getitem__(self, item):
        item = int(item // self.compare_with)
        # find fitting index
        while True:
            idx = self.rng.integers(len(self.states))
            if idx == item:
                continue
            else:
                break

        return np.concatenate((self.states[item], self.states[idx]))