from train_online import train_online
from train_offline import train_offline
from multiprocessing import Pool

"""
Test DQN behavioural policy
"""

# project parameters
envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'LunarLander-v2']
agent_types = ["DQN", "REM"]
multiple_runs = 2
# experiment parameters
experiment = 2
seed = 42
# hyper-parameters for online training
use_random = False
# hyper-parameters for offline training
transitions = 200000
batch_size = 128


def train(envid):
    train_online(experiment=experiment, envid=envid, run=1, seed=seed, use_random=use_random)
    for agent in agent_types:
        for run in range(1, multiple_runs + 1):
            train_offline(experiment=experiment, envid=envid, agent_type=agent,
                          transitions=transitions, batch_size=batch_size, use_run=1, run=run, seed=seed+run)


if __name__ == '__main__':
    with Pool(len(envs)) as p:
        p.map(train, envs)


