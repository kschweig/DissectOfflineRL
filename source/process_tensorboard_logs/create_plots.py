import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

image_type = "png"

# metric manager
with open(os.path.join("..", "..", "data", f"ex5", "metrics.pkl"), "rb") as f:
    mm = pickle.load(f)

# for reward normalisation
envs = {'CartPole-v1':0, 'MountainCar-v0':1, "MiniGrid-LavaGapS6-v0":2, "MiniGrid-SimpleCrossingS9N1-v0":3}
random_rewards = [0, -200, 0, 0]
optimal_rewards = [500, -90, 0.95, 0.961]

####################################
#       Usual Reward plots         #
####################################

mark = "reward"

# titles
y_label = "Normalized Reward"
x_label = "Update Steps"
algos = ["BC", "BVE", "EVMCP", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
buffer = {"er": "Experience Replay", "fully": "Final Policy", "random": "Random Policy",
          "mixed": "Mixed Policy", "noisy": "Noisy Final Policy"}

def plt_csv(csv, algo):
    est = np.mean(csv, axis=1)
    sd = np.std(csv, axis=1)
    cis = (est - sd, est + sd)

    plt.fill_between(np.arange(0, len(est) * 100, 100), cis[0], cis[1], alpha=0.2)
    plt.plot(np.arange(0, len(est) * 100, 100), est, label=algo)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.title(env + " (" + buffer[mode] + ")")


indir = os.path.join("..", "..", "results", "csv", mark)
outdir = os.path.join("..", "..", "results", "img", mark)
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    env = file.split("/")[-1].split("_")[0]
    mode = file.split("_")[1]
    algo = file.split("_")[2].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or mode not in data[env].keys():
        data[env][mode] = dict()

    # normalize reward
    csv -= random_rewards[envs[env]]
    csv /= (optimal_rewards[envs[env]] - random_rewards[envs[env]])

    data[env][mode][algo] = csv

for env in data.keys():
    for mode in data[env].keys():
        if mode == "online":
            continue

        plt.figure(figsize=(8, 6))
        csv = data[env]["online"]["DQN"]

        plt.hlines(y=csv.max(), xmin=0, xmax=200000, color="black", linewidths=2, label="Online")

        norm = (mm.get_data(env, mode)[0][0] - random_rewards[envs[env]] ) / (optimal_rewards[envs[env]] - random_rewards[envs[env]])
        plt.hlines(y=norm, xmin=0, xmax=200000, color="black", linestyles="dotted",
                   linewidths=2, label="Behav.")

        for algo in algos:

            csv = data[env][mode][algo]
            plt_csv(csv, algo)

        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + mode + "." + image_type))
        plt.close()


#############################
#       Value plots         #
#############################


#############################
#        Comparisons        #
#############################

###############
# load data
###############
indir = os.path.join("..", "..", "results", "csv", "reward")
outdir = os.path.join("..", "..", "results", "img", "comp")
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    env = file.split("/")[-1].split("_")[0]
    mode = file.split("_")[1]
    algo = file.split("_")[2].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or mode not in data[env].keys():
        data[env][mode] = dict()

    # normalize reward
    csv -= random_rewards[envs[env]]
    csv /= (optimal_rewards[envs[env]] - random_rewards[envs[env]])

    data[env][mode][algo] = (np.mean(csv, axis=1).max(), np.std(csv, axis=1)[np.argmax(np.mean(csv, axis=1))])

###############
# plot metrics + policy
###############
metrics = {(1,0):"Normalized Reward (mean)", (1,1):"Normalized Reward (std)", (2,0):"Entropy (mean)",
           (2,1):"Entropy (std)", (3,0):"Episode Length (mean)", (3,1):"Episode Length (std)",
           (5,0):"Unique States per Episode (mean)", (5,1):"Unique States per Episode (std)",
           (6,0):"Uniqueness (mean)", (6,1):"Uniqueness (std)", 7:"State Uniqueness"}

annotations = ["(a)", "(b)", "(c)", "(d)", "(e)"]
modes = ["random", "mixed", "er", "noisy", "fully"]

for metric in metrics.keys():
    for env in envs:
        plt.figure(figsize=(8, 6))
        plt.ylim(bottom=-0.05, top=1.05)
        plt.ylabel("Normalized Reward (Dataset)")
        plt.xlabel(metrics[metric])
        plt.title(env)

        for algo in algos:
            x, y = [], []
            for mode in modes:
                if metric == 7:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data[env][mode][algo][0])
            x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]
            plt.plot(x, y, "o-", label=algo)

        x, y = [], []
        for mode in modes:
            if metric == 7:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
            y.append(mm.get_data(env, mode)[1][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        plt.plot(x, y, "o-", linestyle="dotted", label="Behav.", color="black")

        xmax, xmin = 0, 9e9
        for m, mode in enumerate(modes):
            if metric == 7:
                x = mm.get_data(env, mode)[metric]
            else:
                x = mm.get_data(env, mode)[metric[0]][metric[1]]
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            plt.text(x, -0.035, annotations[m], ha="center")

        plt.xlim(right=xmax + (xmax - xmin) * 0.18)

        # Online Policy
        csv = data[env]["online"]["DQN"]
        plt.hlines(y=csv[0], xmin=xmin, xmax=xmax, color="black", linewidths=2, label="Online")

        plt.legend(loc="upper right", fontsize="x-small", markerscale=0.7, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + metrics[metric] + "." + image_type))
        plt.close()

    # plot for modes
    for env in envs:
        plt.figure(figsize=(8, 6))
        plt.ylim(bottom=-0.05, top=1.05)
        plt.ylabel("Normalized Reward (Dataset)")
        plt.xlabel("Buffer Type")
        plt.title(env)

        for algo in algos:
            x, y = [], []
            for m, mode in enumerate(modes):
                x.append(m)
                y.append(data[env][mode][algo][0])
            x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]
            plt.plot(x, y, "o-", label=algo)

        x, y = [], []
        for m, mode in enumerate(modes):
            x.append(m)
            y.append(mm.get_data(env, mode)[1][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        plt.plot(x, y, "o-", linestyle="dotted", label="Behav.", color="black")

        plt.xlim(right=max(x) + (max(x) - min(x)) * 0.18)

        # Online Policy
        csv = data[env]["online"]["DQN"]
        plt.hlines(y=csv[0], xmin=0, xmax=len(modes) - 1, color="black", linewidths=2, label="Online")

        plt.xticks(range(len(modes)), [buffer[m] for m in modes], fontsize="small")

        plt.legend(loc="upper right", fontsize="x-small", markerscale=0.7, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_buffertypes" + "." + image_type))
        plt.close()

