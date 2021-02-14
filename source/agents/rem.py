import numpy as np
import os
import copy
import torch
import torch.nn as nn
from source.agents.agent import Agent


class REM(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 heads=200,
                 seed=None):
        super(REM, self).__init__(obs_space, action_space, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 1e-3

        # discounting factor gamma
        self.discount = 0.95

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 8000

        # Q-Networks
        self.Q = Network(self.obs_space, self.action_space, heads=heads).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Optimization
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)

    def policy(self, state, eval=False):

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q.evaluate(state).cpu().numpy().flatten()
                return q_val.argmax(), q_val.max()
        else:
            return self.rng.integers(self.action_space), np.nan

    def train(self, buffer):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + not_done * self.discount * self.Q_target.forward(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q.forward(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "REM"

    def determinancy(self):
        return round((1-max(self.slope * self.iterations + self.initial_eps, self.end_eps))*100, 2)

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))



class Network(nn.Module):

    def __init__(self, num_state, num_actions, heads):
        super(Network, self).__init__()

        self.rng = np.random.default_rng(seed=num_state)
        self.heads = heads
        self.num_actions = num_actions

        self.fnn = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=128),
            nn.SELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.SELU(),
            nn.Linear(in_features=128, out_features=num_actions*heads)
        )

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        # normalize
        #state = normalize(state)

        alphas = self.rng.normal(size=self.heads)
        alphas /= np.sum(alphas)
        alphas = torch.FloatTensor(alphas).to(state.device)

        return torch.sum(self.fnn(state).reshape(len(state), self.num_actions, self.heads) * alphas.reshape(1, 1,-1), dim=2)

    def evaluate(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        # normalize
        #state = normalize(state)

        return torch.mean(self.fnn(state).reshape(len(state), self.num_actions, self.heads), dim=2)






"""

class Network(nn.Module):

    def __init__(self, num_state, num_actions, duelling=False):
        super(Network, self).__init__()

        self.duelling = duelling

        self.shared_stream = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=2*num_state),
            nn.SELU(),
            nn.Linear(in_features=2*num_state, out_features=num_state),
            nn.SELU(),
        )

        self.advantage_stream = nn.Linear(in_features=num_state, out_features=num_actions)
        self.value_stream = nn.Linear(in_features=num_state, out_features=1)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        # normalize
        state = normalize(state)

        features = self.shared_stream(state)

        # if duelling, calculate value and advantage
        if self.duelling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)

            return value + (advantages - advantages.mean())

        # if not, simple calculate values via advantage stream
        else:
            return self.advantage_stream(features)
            
"""