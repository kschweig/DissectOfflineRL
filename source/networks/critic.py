import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, num_state, num_actions, seed, n_estimates=1):
        super(Critic, self).__init__()

        # set seed
        torch.manual_seed(seed)

        self.num_actions = num_actions
        num_hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU()
        )

        self.out = nn.Linear(in_features=num_hidden, out_features=num_actions * n_estimates)

        for param in self.parameters():
            if len(param.shape) > 1:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        state = self.backbone(state)

        return self.out(state)

    def evaluate(self, state):
        return self.forward(state)


class RemCritic(Critic):

    def __init__(self, num_state, num_actions, seed, heads):
        super(RemCritic, self).__init__(num_state, num_actions, seed, heads)

        self.heads = heads

    def forward(self, state):
        state = super(RemCritic, self).forward(state)

        alphas = torch.rand(self.heads).to(device=state.device)
        alphas /= torch.sum(alphas)

        return torch.sum(state.view(len(state), self.heads, self.num_actions) * alphas.view(1, -1, 1), dim=1)

    def evaluate(self, state):
        state = super(RemCritic, self).forward(state)

        return torch.mean(state.view(len(state), self.heads, self.num_actions), dim=1)


class QrCritic(Critic):

    def __init__(self, num_state, num_actions, seed, quantiles):
        super(QrCritic, self).__init__(num_state, num_actions, seed, quantiles)

        self.quantiles = quantiles

    def forward(self, state):
        state = super(QrCritic, self).forward(state)

        return state.reshape(len(state), self.num_actions, self.quantiles)

    def evaluate(self, state):
        state = self.forward(state)

        return torch.mean(state, dim=2)
