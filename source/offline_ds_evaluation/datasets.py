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
        self.rewards = rewards
        self.not_dones = np.invert(dones)

    def __len__(self):
        return len(self.states) - 1

    def __getitem__(self, item):
        return torch.FloatTensor(self.states[item]), torch.FloatTensor(self.states[item+1]), \
               torch.LongTensor(self.actions[item]), torch.LongTensor(self.actions[item+1]), \
               torch.FloatTensor(self.rewards[item]), torch.FloatTensor(self.not_dones[item])


class SCSet(Dataset):

    def __init__(self, states, negative_samples=10, sparse_state=False):
        super(SCSet, self).__init__()

        self.states = states
        self.negative_samples = 1 - (1 / (negative_samples + 1)) if negative_samples >= 0 else -1
        self.sparse_state = sparse_state
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
                    equal = cosine_similarity(self.states[item], self.states[idx]) > 0.95
            else:
                idx = item
                while idx == item:
                    idx = self.rng.integers(self.__len__())

            return np.concatenate((self.states[item], self.states[idx])), np.zeros((1), dtype=np.float32)

        return np.concatenate((self.states[item], self.states[item])), np.ones((1), dtype=np.float32)


class StartSet(Dataset):

    def __init__(self, states, dones, subsample=1):
        super(StartSet, self).__init__()

        assert subsample >= 0 and subsample <= 1, f"subsampling must be in [0,1], is {subsample}"

        self.states = []
        self.dones = dones
        self.subsample = subsample

        self.states = states[np.where(dones == 1)[0]]

    def __len__(self):
        return int(len(self.states)**2 * self.subsample)

    def __getitem__(self, item):
        length = int(np.sqrt(self.__len__() / self.subsample))
        return np.concatenate((self.states[item // length], self.states[item % length]))


class NSCSet(Dataset):

    def __init__(self, states, actions, dones, num_actions, negative_samples=10, sparse_state=False):
        super(NSCSet, self).__init__()

        self.states = states
        self.actions = actions
        self.dones = dones
        self.num_actions = num_actions
        self.negative_samples = 1 - (1 / (negative_samples + 1)) if negative_samples >= 0 else -1
        self.sparse_state = sparse_state
        self.rng = np.random.default_rng(seed=42)

    def __len__(self):
        return len(self.states) - 1

    def __getitem__(self, item):

        action = np.zeros((self.num_actions))
        action[self.actions[item]] = 1

        if self.rng.random() < self.negative_samples or self.negative_samples < 0:
            if self.sparse_state:
                equal = True
                while equal:
                    idx = self.rng.integers(self.__len__())
                    if idx == (item + 1):
                        continue
                    equal = cosine_similarity(self.states[item + 1], self.states[idx]) > 0.95
            else:
                idx = item + 1
                while idx == (item + 1):
                    idx = self.rng.integers(self.__len__())
            return np.concatenate((self.states[item], action, self.states[idx] * self.dones[idx]), dtype=np.float32), \
                   np.zeros((1), dtype=np.float32)

        return np.concatenate((self.states[item], action, self.states[item+1] * self.dones[item+1]), dtype=np.float32), \
               np.ones((1), dtype=np.float32)
