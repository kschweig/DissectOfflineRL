import os
import pickle
import numpy as np
from source.train_online import train_online
from source.train_offline import train_offline
from source.offline_ds_evaluation.evaluator import Evaluator
from multiprocessing import Pool

"""
Test Random behavioural policy
"""

# project parameters
envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
discounts = [0.9, 0.99, 0.95]
agent_types = ["DQN"]
#agent_types = ["BC", "SAC", "BCQ", "DQN", "QRDQN"]
multiple_runs = 2
# experiment parameters
experiment = 1
seed = 42
# hyperparameters for online training
behavioral = "DQN"
# hyperparameters for offline training
transitions = 200000
batch_size = 128


def train(args):
    envid, discount = args
    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid,
                 transitions=transitions, buffer_size=50000,
                 run=1, seed=seed)

    """
    for agent in agent_types:
        for run in range(1, multiple_runs + 1):
            train_offline(experiment=experiment, envid=envid, agent_type=agent, discount=discount,
                          transitions=transitions, batch_size=batch_size, use_run=1, run=run, seed=seed+run)
    """

def assess_ds(args):
    envid, use_run = args

    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{use_run}.pkl"), "rb") as f:
        buffer = pickle.load(f)

    eval = Evaluator(buffer.state, buffer.action, buffer.reward, np.invert(buffer.not_done))
    print(envid, " Rewards:", eval.get_rewards())
    #eval.train_behavior_policy(batch_size=256, epochs=2)
    #print(envid," Behavioral Entropy:", eval.get_bc_entropy())
    #eval.train_value_critic(batch_size=256, epochs=3, lr=1e-3, horizon=500, verbose=True)
    #print(envid," Behavioral Value:", eval.get_value_estimate())
    #eval.train_state_comparator(batch_size=256, epochs=5)

    #print("same states")
    #test = eval.test_state_compare(negative_samples=0)
    #print(np.min(test), np.mean(test), np.max(test))
    #print("disjoint states")
    #test = eval.test_state_compare(negative_samples=-1)
    #print(np.min(test), np.mean(test), np.max(test))

    #print(eval.get_start_randomness())
    #print(eval.get_unique_pathlength())


    eval.train_next_state_comparator(batch_size=256, epochs=3, negative_samples=1, sparse_state=False, verbose=True)
    test = eval.test_next_state_compare(negative_samples=0, sparse_state=True)
    print(np.min(test), np.mean(test), np.max(test))
    test=eval.test_next_state_compare(negative_samples=-1, sparse_state=True)
    print(np.min(test), np.mean(test), np.max(test))


if __name__ == '__main__':
    #with Pool(len(envs), maxtasksperchild=1) as p:
        #p.map(train, zip(envs,discounts))
        #p.map(assess_ds, zip(envs, [1]*3))

    assess_ds((envs[0], 1))
    #assess_ds((envs[1], 1))
    #assess_ds((envs[2], 1))