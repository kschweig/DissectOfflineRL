import gym
from gym import spaces
import numpy as np


class MinAtarObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * obs_shape[1], ),
            dtype='uint8'
        )

    def observation(self, obs):
        # division by 10 as the first dimension can hold up to 5 different colors
        # and the second channel can hold up to 10 different objects
        # -0.5 to center
        channels = obs.shape[2]
        obs = obs * np.arange(1, channels + 1)[None, None, :]
        return (np.max(obs, axis=2).flatten() / channels) - 0.5


class FlatImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission and flatten it
    Modified from https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py
    to not only remove mission string but also remove the third dimension(representing the door state that can be open,
    closed or locked) which is unnecessary as I am not using doors and flatten from 7x7x2 to 98.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.spaces['image'].shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * obs_shape[1] * 2, ),
            dtype='uint8'
        )

    def observation(self, obs):
        # division by 10 as the first dimension can hold up to 5 different colors
        # and the second channel can hold up to 10 different objects
        # -0.5 to center
        return (obs['image'][:,:,:2].flatten() / 10) - 0.5


class RestrictMiniGridActionWrapper(gym.core.ActionWrapper):
    """
    restrict to the first three actions -> turn left, turn right and move forward.
    This is sufficient for the used environments from MiniGrid
    """
    def __init__(self, env):
        super(RestrictMiniGridActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action

class RestrictMinAtarActionWraper(gym.core.ActionWrapper):
    """
    restrict to the minimal actions for the respective minatar environment
    """
    def __init__(self, env):
        super(RestrictMinAtarActionWraper, self).__init__(env)

        self.action_space = gym.spaces.Discrete(len(env.game.env.minimal_action_set()))
        env.game.env.action_map = [env.game.env.action_map[i] for i in env.game.env.minimal_action_set()]
        # print(env.game_name, env.game.env.action_map)

    def action(self, action):
        return action


class RewardAtEndWrapper(gym.core.RewardWrapper):
    """
    Wrapper for Acrobot and MountainCar to change reward from negative reward throughout training
    to a single positive reward upon reaching the goal.
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # only give a single reward upon episode end.
        return observation, done and not self.env._elapsed_steps == self.env._max_episode_steps, done, info
