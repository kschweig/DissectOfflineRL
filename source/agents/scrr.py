
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import Critic
from ..networks.actor import Actor


class SCRR(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 seed=None):
        super(SCRR, self).__init__(obs_space, action_space, 1, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # Optimization
        self.lr = 1e-4

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)

        # policy network
        self.actor = Actor(self.obs_space, self.action_space, seed).to(self.device)
        self.p_optim = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

        self.beta = 1


    def policy(self, state, eval=False):

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q.evaluate(state).cpu()

                actions = self.actor(state).cpu()
                actions = F.softmax(actions, dim=1)
                dist = Categorical(actions.unsqueeze(0))

                return dist.sample().item(), q_val.max().item(), entropy(actions)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas, use_remaining_reward=True)

        # log state distribution
        if self.iterations % 1000 == 0:
            writer.add_histogram("train/states", state, self.iterations)

        # Get current Q estimate
        current_Qs = self.Q.forward(state)
        current_Q = current_Qs.gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, reward)

        # log temporal difference error
        writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        # calculate advantage
        with torch.no_grad():
            advantage = current_Qs - current_Qs.mean(dim=1, keepdim=True)

        # predict action the behavioral policy would take
        pred_action = self.actor.forward(state)
        log_actions = F.log_softmax(pred_action, dim=1)

        # policy loss
        # exp style
        loss = -(log_actions * torch.exp(advantage / self.beta)).sum(dim=1).mean()
        # binary style
        #loss = -(log_actions * torch.heaviside(advantage, values=torch.zeros(1).to(self.device))).sum(dim=1).mean()

        writer.add_scalar("train/policy-loss", torch.mean(loss).detach().cpu().item(), self.iterations)

        # optimize policy
        self.p_optim.zero_grad()
        loss.backward()
        self.p_optim.step()

        self.iterations += 1

    def get_name(self) -> str:
        return "SCRR"

    def determinancy(self):
        return round((1-max(self.slope * self.iterations + self.initial_eps, self.end_eps))*100, 2)

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))