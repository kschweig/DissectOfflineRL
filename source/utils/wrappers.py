import gym
from gym import spaces


class FlatImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission and flatten it
    Modified from https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py
    to not only remove mission string but also flatten from 7x7x3 to 147.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.spaces['image'].shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * obs_shape[1] * obs_shape[2], ),
            dtype='uint8'
        )

    def observation(self, obs):
        return obs['image'].flatten()