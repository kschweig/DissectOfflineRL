from source.train_online import train_online
from source.train_offline import train_offline
from multiprocessing import Pool
from source.utils.evaluation import evaluate
from source.utils.utils import make_env
import os
from torch.utils.tensorboard import SummaryWriter

"""
Test Batch Constrained Q-learning
"""

# project parameters
agent_types = ["SQN"]
multiple_runs = 2
# experiment parameters
experiment = 3
seed = 42
# hyperparameters for online training
behavioral = "DQN"
transitions_online = 400000
# hyperparameters for offline training
transitions_offline = 200000
batch_size = 128


def train(args):
    envid, discount = args
    """
    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid,
                 transitions=transitions_online, buffer_size=50000,
                 run=1, seed=seed)
    """
    for agent in agent_types:
        for run in range(1, multiple_runs + 1):
            ag = train_offline(experiment=experiment, envid=envid, agent_type=agent, discount=discount,
                          transitions=transitions_offline, batch_size=batch_size, use_run=1, run=run, seed=seed+run, use_remaining_reward=True)

            # test on different version of env
            env = make_env('MiniGrid-DistShift2-v0')
            writer = SummaryWriter(log_dir=os.path.join("runs", f"ex{experiment}", f"MiniGrid-DistShift2-v0_{agent}_run{run}"))
            all_rewards = []
            for i in range(10000):
                all_rewards = evaluate(env, ag, writer, all_rewards, over_episodes=100)



if __name__ == '__main__':
    train(('MiniGrid-DistShift1-v0', 0.95))