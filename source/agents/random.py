from source.agents.agent import Agent
import numpy as np

class Random(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 seed=None):
        super(Random, self).__init__(obs_space, action_space, seed)

    def get_name(self) -> str:
        return "Random"

    def policy(self, state, available, eval=False):
        possible = np.arange(1, self.action_space+1)
        possible * available
        possible = np.asarray([i - 1 for i in possible[available != 0]])

        return self.rng.choice(possible), np.nan

    def train(self, buffer):
        pass

    def determinancy(self):
        return 0.0

    def save_state(self) -> None:
        pass

    def load_state(self) -> None:
        pass