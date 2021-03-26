import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, num_state, num_actions):
        super(Actor, self).__init__()

        num_hidden = 256

        self.fnn = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_actions)
        )

        for param in self.fnn.parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        return self.fnn(state)

    def evaluate(self, state):
        return self.forward(state)