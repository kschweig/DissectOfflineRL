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


class SQN(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 seed=None):
        super(SQN, self).__init__(obs_space, action_space, discount, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()
        self.nll = nn.NLLLoss()

        # threshold for actions, unlikely under the behavioral policy
        self.threshold = 0.3

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 1

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        # BCQ has a separate Actor, as I have no common vision network
        self.actor = Actor(self.obs_space, self.action_space, seed).to(self.device)

        # Optimization
        self.lr = 1e-4
        self.Q_optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)

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

                # masking non-eligible values with -9e9 to be sure they are not sampled
                return q_val.argmax(dim=1).item(), \
                       q_val.max().item(), entropy(q_val)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):

        # calculate buffer similarities only once
        if self.iterations == 0:
            buffer.calc_norm(include_actions=True)

        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas)

        # log state distribution
        if self.iterations % 1000 == 0:
            writer.add_histogram("train/states", state, self.iterations)

        with torch.no_grad():
            q_val = self.Q.forward(next_state)
            actions = self.actor.forward(next_state)

            sm = F.softmax(actions, dim=1)

            probas = torch.zeros((self.action_space)).to(self.device)

            # if within dataset, behavioral cloning policy defines probability to sample
            for a in range(self.action_space):
                _, sim = buffer.get_closest((next_state + a).cpu().numpy())
                if sim > 0.95:
                    probas[a] = sm[a]
                    q_val[a] -= 9e9 # mask out

            # highest q_value gets remaining probability
            probas[q_val.argmax(dim=1).item()] = 1. - probas.sum().item()
            dist = Categorical(probas)

            q_val = self.Q_target(next_state)
            target_Q = reward + not_done * self.discount * self.Q_target.forward(next_state).gather(1, dist.sample().item())

        # Get current Q estimate and actor decisions on actions
        current_Q = self.Q.forward(state).gather(1, action)
        actions = self.actor.forward(state)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)
        A_loss = self.nll(F.log_softmax(actions, dim=1), action.reshape(-1))

        # log temporal difference error
        writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)
        writer.add_scalar("train/CE-loss", torch.mean(A_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        A_loss.backward()
        self.actor_optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "SPIBB"

    def determinancy(self):
        return round((1-max(self.slope * self.iterations + self.initial_eps, self.end_eps))*100, 2)

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))