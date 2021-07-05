import torch
import torch.nn as nn
from abc import ABC


class BaseNet(nn.Module, ABC):

    def __init__(self, num_state, seed):
        super(BaseNet, self).__init__()

        # set seed
        torch.manual_seed(seed)

        self.num_hidden = 256

        self.base = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.SELU()
        )

        for param in self.parameters():
            if len(param.shape) == 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        return self.base(state)


class BC(BaseNet):

    def __init__(self, num_state, num_actions, seed):
        super(BC, self).__init__(num_state, seed)

        self.out = nn.Linear(in_features=self.num_hidden, out_features=num_actions)

        for param in self.out.parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        state = super(BC, self).forward(state)

        return self.out(state)


class Embedding(nn.Module):

    def __init__(self, num_state, num_embedding, seed):
        super(Embedding, self).__init__()

        # set seed
        torch.manual_seed(seed)

        self.num_hidden = 256

        self.net = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=num_embedding))

        for param in self.parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def embed(self, state):
        return self.net(state)

    def forward(self, states):
        (s1, s2) = states

        embedding_1 = self.embed(s1)
        embedding_2 = self.embed(s2)

        # calculate cosine similarities between embeddings -> (-1, 1)
        out = torch.diag(embedding_1 @ embedding_2.T, diagonal=0) / (torch.linalg.norm(embedding_1, dim=1) *
                                                                     torch.linalg.norm(embedding_2, dim=1))

        # change output range to (0, 1) with sigmoid to be applicable to bceloss
        out = (out + 1.) / 2.

        return torch.sigmoid(out)

