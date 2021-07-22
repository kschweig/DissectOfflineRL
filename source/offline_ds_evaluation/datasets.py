import numpy as np
import torch
from torch.utils.data import Dataset


class BCSet(Dataset):

    def __init__(self, states, actions):
        super(BCSet, self).__init__()

        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return torch.FloatTensor(self.states[item]), torch.LongTensor(self.actions[item])


class StateActionEmbeddingSet(Dataset):

    def __init__(self, states, dones, seed, actions=None, negative_sampling=1):
        super(StateActionEmbeddingSet, self).__init__()

        # add action to embedding if not none
        self.states = states + (actions if actions is not None else 0)
        self.dones = dones

        self.negative_sampling = negative_sampling
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (self.negative_sampling + 1) * len(self.states)

    def __getitem__(self, item):

        if item < len(self.states) - 1:
            if not self.dones[item]:
                return (self.states[item], self.states[item + 1]), np.array(1, dtype=np.float32)
            else:
                return (self.states[item], self.states[item + 1]), np.array(0, dtype=np.float32)

        # sample some random index if we use a negative sample.
        item = item % len(self.states)
        idx = self.rng.integers(len(self.states))
        # in case the same index or the correct one is sampled by chance
        while idx == item or idx == item + 1:
            idx = self.rng.integers(len(self.states))

        return (self.states[item], self.states[idx]), np.array(0, dtype=np.float32)


class AESet(Dataset):
    def __init__(self, states, actions=None):
        super(AESet, self).__init__()

        self.states = states + (actions if actions is not None else 0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return self.states[item], self.states[item]
