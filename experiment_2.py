from train_online import train_online
from train_offline import train_offline
from multiprocessing import Pool

"""
Test DQN behavioural policy
"""

# project parameters
envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'MiniGrid-Empty-Random-6x6-v0']
discounts = [0.9, 0.99, 0.95, 0.95]
agent_types = ["DQN", "REM"]
multiple_runs = 2
# experiment parameters
experiment = 2
seed = 42
# hyperparameters for online training
behavioral = "DQN"
# hyperparameters for offline training
transitions = 200000
batch_size = 128


def train(args):
    envid, discount = args
    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid, run=1, seed=seed)
    for agent in agent_types:
        for run in range(1, multiple_runs + 1):
            train_offline(experiment=experiment, envid=envid, agent_type=agent, discount=discount,
                          transitions=transitions, batch_size=batch_size, use_run=1, run=run, seed=seed+run)


if __name__ == '__main__':
    with Pool(len(envs), maxtasksperchild=1) as p:
        p.map(train, zip(envs,discounts))