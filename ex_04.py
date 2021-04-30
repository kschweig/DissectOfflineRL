from source.train_online import train_online
from source.train_offline import train_offline
from multiprocessing import Pool

"""
Test Batch Constrained Q-learning
"""

# project parameters
envs = ['Acrobot-v1', 'MountainCar-v0']
discounts = [0.95, 0.99]
buffer_types = ["er", "fully", "random"]
agent_types = ["BC", "SQN", "DQN", "QRDQN", "SAC", "BCQ", "REM", "CRR"]
multiple_runs = 1
# experiment parameters
experiment = 4
seed = 42
# hyperparameters for online training
behavioral = "DQN"
transitions_online = 100000
# hyperparameters for offline training
transitions_offline = 100000
batch_size = 128


def train(args):
    envid, discount = args

    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid,
                 transitions=transitions_online, buffer_size=50000,
                 run=1, seed=seed)

    for agent in agent_types:
        if agent == "SQN":
            for run in range(1, multiple_runs + 1):
                for bt in range(len(buffer_types)):
                    train_offline(experiment=experiment, envid=envid, agent_type=agent, buffer_type=buffer_types[bt],
                                  discount=discount, transitions=transitions_offline, batch_size=batch_size, use_run=1,
                                  run=run, seed=seed+run, use_remaining_reward=True)
        else:
            for run in range(1, multiple_runs + 1):
                for bt in range(len(buffer_types)):
                    train_offline(experiment=experiment, envid=envid, agent_type=agent, buffer_type=buffer_types[bt],
                                  discount=discount, transitions=transitions_offline, batch_size=batch_size, use_run=1,
                                  run=run, seed=seed + run)

if __name__ == '__main__':
    with Pool(len(envs), maxtasksperchild=1) as p:
        p.map(train, zip(envs, discounts))