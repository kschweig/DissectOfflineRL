from source.train_online import train_online
from source.train_offline import train_offline
from source.offline_ds_evaluation.evaluator import Evaluator
from source.offline_ds_evaluation.metrics_manager import MetricsManager
from source.offline_ds_evaluation.latex import create_latex_table
from multiprocessing import Pool
import os
import pickle
import numpy as np


# project parameters
envs = ['Breakout-MinAtar-v0', "Space_invaders-MinAtar-v0"]
discounts = [0.99, 0.99]
buffer_types = ["random", "mixed", "er", "noisy", "fully"]
agent_types = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
multiple_runs = 3
# experiment parameters
experiment = 4
seed = 42
# hyperparameters for online training
behavioral = "DQN"
transitions_online = 100000
# hyperparameters for offline training
transitions_offline = 2 * transitions_online
batch_size = 128
lr = [1e-4] * len(agent_types)
# parameters for evaluation
random_rewards = [0, 0]
optimal_rewards = [10, 20]

def create_ds(args):
    envid, discount = args

    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid,
                 transitions=transitions_online, buffer_size=50000,
                 run=1, seed=seed)

def train(args):
    envid, discount = args

    for run in range(1, multiple_runs + 1):
        for a, agent in enumerate(agent_types):
            for bt in range(len(buffer_types)):
                train_offline(experiment=experiment, envid=envid, agent_type=agent, buffer_type=buffer_types[bt],
                              discount=discount, transitions=transitions_offline, batch_size=batch_size, lr=lr[a],
                              use_run=1, run=run, seed=seed+run, use_remaining_reward=(agent == "MCE"))

def assess_env(args):
    e, envid = args

    os.makedirs(os.path.join("results", "ds_eval"), exist_ok=True)

    results = []
    mm = MetricsManager(experiment)
    for buffer_type in buffer_types:
        with open(os.path.join("data", f"ex{experiment}", f"{envid}_run1_{buffer_type}.pkl"), "rb") as f:
            buffer = pickle.load(f)

        evaluator = Evaluator(envid, buffer_type, buffer.state, buffer.action, buffer.reward,
                              np.invert(buffer.not_done))

        path = os.path.join("results", "ds_eval", f"{envid}_{buffer_type}")
        data = evaluator.evaluate(path, random_rewards[e], optimal_rewards[e], epochs=10)

        results.append(data)
        mm.append(data)

    for i in range(0, len(buffer_types)):
        texpath = os.path.join("results", "ds_eval", f"{results[i][0]}.tex")
        print(texpath)

    create_latex_table(texpath, results)

    with open(os.path.join("data", f"ex{experiment}", f"metrics_{envid}.pkl"), "wb") as f:
        pickle.dump(mm, f)


if __name__ == '__main__':

    with Pool(len(envs), maxtasksperchild=1) as p:
        #p.map(create_ds, zip(envs, discounts))
        #p.map(train, zip(envs, discounts))
        p.map(assess_env, zip(range(len(envs)), envs))
