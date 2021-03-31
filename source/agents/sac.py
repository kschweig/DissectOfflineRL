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


class SAC(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 seed=None):
        super(SAC, self).__init__(obs_space, action_space, discount, seed)

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

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 1

        # Explicit Policy
        self.actor = Actor(self.obs_space, self.action_space, seed).to(self.device)
        # Q-Networks
        self.Q1 = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2 = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q2_target = copy.deepcopy(self.Q2)

        # Entropy target
        self.target_entropy = -np.log((1.0 / self.action_space)) * 0.98

        # Alpha tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # soft update
        self.tau = 0.005

        # Optimization
        self.lr = 1e-4
        self.Q_optimizer = torch.optim.Adam(params=list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=self.lr)

    def policy(self, state, eval=False):

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                actions = self.actor.evaluate(state).cpu()
                actions = F.softmax(actions, dim=1)
                q1_vals = self.Q1.evaluate(state).cpu()
                q2_vals = self.Q2.evaluate(state).cpu()
                if eval == True:
                    action = actions.argmax().item()
                    return action, torch.min(q1_vals[0,action], q2_vals[0,action]), entropy(actions)
                else:
                    dist = Categorical(actions.squeeze(0))
                    return dist.sample().item(), np.nan, entropy(actions)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas)

        # log state distribution
        if self.iterations % 1000 == 0:
            writer.add_histogram("train/states", state, self.iterations)

        # Compute the target Q value
        with torch.no_grad():
            target_Q = torch.min(self.Q1_target.forward(next_state), self.Q2_target.forward(next_state))

            log_action_probs = F.log_softmax(self.actor.forward(state), dim=1)
            action_probs = log_action_probs.exp()

            target_Q = action_probs * (target_Q - self.alpha * log_action_probs)

            target_Q = reward + not_done * self.discount * target_Q.sum(dim=1, keepdim=True)

        # Get current Q estimate
        current_Q1 = self.Q1.forward(state).gather(1, action)
        current_Q2 = self.Q2.forward(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q1_loss = self.huber(current_Q1, target_Q)
        Q2_loss = self.huber(current_Q2, target_Q)

        # log temporal difference error
        writer.add_scalar("train/TD-error 1", torch.mean(Q1_loss).detach().cpu().item(), self.iterations)
        writer.add_scalar("train/TD-error 2", torch.mean(Q2_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q's
        self.Q_optimizer.zero_grad()
        Q1_loss.backward()
        Q2_loss.backward()
        self.Q_optimizer.step()

        # compute target Q for actor
        with torch.no_grad():
            target_Q = torch.min(self.Q1_target.forward((next_state)), self.Q2_target.forward((next_state)))

        # Compute policy loss
        log_action_probs = F.log_softmax(self.actor.forward(state), dim=1)
        action_probs = log_action_probs.exp()

        policy_loss = action_probs * (self.alpha.detach() * log_action_probs - target_Q)
        policy_loss = policy_loss.sum(dim=1).mean()

        # log policy loss
        writer.add_scalar("train/policy-loss", torch.mean(policy_loss).detach().cpu().item(), self.iterations)

        # Optimize the policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


        # compute alpha loss
        with torch.no_grad():
            log_action_probs = F.log_softmax(self.actor.forward(state), dim=1)
            action_probs = log_action_probs.exp()

        alpha_loss = -self.alpha * (log_action_probs + self.target_entropy)
        alpha_loss = alpha_loss.sum(dim=1).mean()

        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # update alpha
        self.alpha = self.log_alpha.exp()

        # log alpha loss
        writer.add_scalar("train/alpha", self.alpha.detach().cpu().item(), self.iterations)
        writer.add_scalar("train/alpha-loss", torch.mean(alpha_loss).detach().cpu().item(), self.iterations)



        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_name(self) -> str:
        return "SAC"

    def determinancy(self):
        return round((1-max(self.slope * self.iterations + self.initial_eps, self.end_eps))*100, 2)

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))