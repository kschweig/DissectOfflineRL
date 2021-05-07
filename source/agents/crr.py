import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import Critic
from ..networks.actor import Actor


class CRR(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr=1e-4,
                 seed=None):
        super(CRR, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # policy network
        self.actor = Actor(self.obs_space, self.action_space, seed).to(self.device)

        # huber loss
        self.huber = nn.SmoothL1Loss()

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)
        self.p_optim = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

        # Temperature parameter
        self.beta = 1

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q(state).cpu()

                actions = self.actor(state).cpu()
                actions = F.softmax(actions, dim=1)
                dist = Categorical(actions.unsqueeze(0))

                #return q_val.argmax().item(), q_val, entropy(actions)
                return dist.sample().item(), q_val, entropy(actions)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas)

        ###################################
        ### Policy update
        ###################################

        # set networks to train mode
        self.actor.train()
        self.Q.train()
        self.Q_target.train()

        # calculate advantage
        with torch.no_grad():
            current_Qs = self.Q(state)
            advantage = current_Qs - current_Qs.mean(dim=1, keepdim=True)

        # predict action the policy would take
        log_actions = F.log_softmax(self.actor(state), dim=1)

        # policy loss
        # exp style
        #loss = -(log_actions * torch.exp(advantage / self.beta)).sum(dim=1).mean()
        # binary style
        loss = -(log_actions * torch.heaviside(advantage, values=torch.zeros(1).to(self.device))).sum(dim=1).mean()

        writer.add_scalar("train/policy-loss", torch.mean(loss).detach().cpu().item(), self.iterations)

        # optimize policy
        self.p_optim.zero_grad()
        loss.backward()
        self.p_optim.step()

        ###################################
        ### Critic update
        ###################################

        # set networks to train mode
        self.actor.train()
        self.Q.train()
        self.Q_target.train()

        # Compute the target Q value
        with torch.no_grad():
            actions = self.actor(next_state)
            actions = F.log_softmax(actions, dim=1)
            dist = Categorical(logits=actions)
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).gather(1, dist.sample().unsqueeze(1))

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # log temporal difference error
        writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "CriticRegularizedRegression"

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.actor.state_dict(), os.path.join("models", self.get_name() + "_actor.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim1.pt"))
        torch.save(self.p_optim.state_dict(), os.path.join("models", self.get_name() + "_optim2.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.actor.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_actor.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim1.pt")))
        self.p_optim.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim2.pt")))