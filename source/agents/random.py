from source.agents.agent import Agent
import numpy as np
from source.evaluation import entropy
import torch


class Random(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 seed=None):
        super(Random, self).__init__(obs_space, action_space, discount, seed)

    def get_name(self) -> str:
        return "Random"

    def policy(self, state, eval=False):
        possible = np.arange(self.action_space)

        return self.rng.choice(possible), np.nan, \
               entropy((torch.ones(self.action_space) / self.action_space).float().view(1,-1))

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        pass

    def determinancy(self):
        return 0.0

    def save_state(self) -> None:
        pass

    def load_state(self) -> None:
        pass