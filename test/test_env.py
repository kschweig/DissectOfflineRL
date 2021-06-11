import gym
import gym_minigrid
import gym_minatar
from gym_minigrid.wrappers import FullyObsWrapper
from gym_minigrid.window import Window
from source.utils.wrappers import FlatImgObsWrapper, MinAtarObsWrapper

"""
CartPole-v1:
max transitions: 500, reward: 1 per timestep if not done,
max score: 500, solved at >= 475 over 100 episodes

Acrobot-v1:
max transitions: 500 , reward: -1 per timestep if not done,
max score: as high as possible, no bound given for solving, I achieved ~-85

MountainCar-v0:
max transitions: 200, reward: -1 per timestep if not done,
max score: as high as possible, solved at >= -110 over 100 episodes

LunarLander-v2:
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points. Episode until touching the ground.

MiniGrid-LavaGapS6-v0:
0 reward if connect with lava, 1 - 0.9 * (self.step_count / self.max_steps) if the end goal is reached.
max_steps = 144
min_steps = 7(if good env with gap on top)/8
--> max_reward = 0.95625 / 0.94375s

MiniGrid-SimpleCrossingS9N1-v0:
1 - 0.9 * (self.step_count / self.max_steps) if the end goal is reached, 0 if not.
max_steps = 324
min_steps = 14
--> max_reward = 0.961
"""
envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']

#env = gym.make(envs[2])
env = (gym.make("Breakout-MinAtar-v0"))
env = MinAtarObsWrapper(gym.make("Space_invaders-MinAtar-v0"))

#env = FlatImgObsWrapper(FullyObsWrapper(gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')))
#env = FlatImgObsWrapper(gym.make('MiniGrid-LavaGapS6-v0'))
obs = env.reset()
print(obs.shape)
print(env.action_space.n)
print(obs)
#window = Window(title="MiniGrid")
for _ in range(10):
    #img = env.render('rgb_array')
    #window.show_img(img)
    obs, _, _, _ = env.step(env.action_space.sample()) # take a random action
env.close()