import os
import glob
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

image_type = "png"
figsize = (7, 5)

# metric manager
experiments = ["ex4", "ex5", "ex6"]

mm = MetricsManager(0)
for ex in experiments:
    with open(os.path.join("..", "..", "data", ex, "metrics.pkl"), "rb") as f:
        m = pickle.load(f)
    mm.data.update(m.data)

# static stuff

envs = {'CartPole-v1':0, 'MountainCar-v0':1, "MiniGrid-LavaGapS7-v0":2, "MiniGrid-Dynamic-Obstacles-8x8-v0":3,
        'Breakout-MinAtar-v0': 4, "Space_invaders-MinAtar-v0": 5}

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
#algos = ["BC", "MCE", "DQN", "BCQ", "CQL"]

buffer = {"er": "Experience Replay", "fully": "Final Policy", "random": "Random Policy",
          "mixed": "Mixed Policy", "noisy": "Noisy Policy"}

random_rewards = [0, -200, 0, -1, 0, 0]
optimal_rewards = [500, -90, 0.95, 0.94, 8, 20]

y_bounds = {'CartPole-v1': (-10, 15), "MiniGrid-LavaGapS7-v0":(-1, 1), 'MountainCar-v0': (-10, 15),
            "MiniGrid-Dynamic-Obstacles-8x8-v0":(-1, 1), 'Breakout-MinAtar-v0': (-10,50), "Space_invaders-MinAtar-v0": (-10,50)}

def plt_csv(csv, algo, set_title=True, color=None):
    est = np.mean(csv, axis=1)
    sd = np.std(csv, axis=1)
    cis = (est - sd, est + sd)

    plt.fill_between(np.arange(0, len(est) * 100, 100), cis[0], cis[1], alpha=0.2, color=color)
    plt.plot(np.arange(0, len(est) * 100, 100), est, label=algo, color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if set_title:
        plt.title(env + " (" + buffer[mode] + ")")


####################################
#       Usual Return plots         #
####################################

mark = "return"

# titles
y_label = "Normalized Return"
x_label = "Update Steps"

indir = os.path.join("..", "..", "results", "csv", mark)
outdir = os.path.join("..", "..", "results", "img", mark)
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

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

        plt.figure(figsize=figsize)
        csv = data[env]["online"]["DQN"]
        plt.hlines(y=csv.max(), xmin=0, xmax=200000, color="black", linewidths=2)
        plt_csv(csv, "Online", color="black")

        norm = (mm.get_data(env, mode)[0][0] - random_rewards[envs[env]] ) / (optimal_rewards[envs[env]] - random_rewards[envs[env]])
        plt.hlines(y=norm, xmin=0, xmax=200000, color="black", linestyles="dotted",
                   linewidths=2, label="Behav.")

        for a, algo in enumerate(algos):
            csv = data[env][mode][algo]
            plt_csv(csv, algo, color=f"C{a}")

        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + mode + "." + image_type))
        plt.close()

##################################
#    Action-Value Deviations     #
##################################

mark = "action_value_deviation"

# titles
y_label = "Action-Value deviation"
x_label = "Update Steps"

indir = os.path.join("..", "..", "results", "csv", mark)
outdir = os.path.join("..", "..", "results", "img", mark)
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)
data_avd = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    if not data_avd.keys() or env not in data_avd.keys():
        data_avd[env] = dict()
    if not data_avd[env].keys() or mode not in data_avd[env].keys():
        data_avd[env][mode] = dict()

    data_avd[env][mode][algo] = csv

algos_ = algos.copy()
algos_.remove("BC")
for env in data_avd.keys():
    for mode in data_avd[env].keys():
        if mode == "online":
            continue

        plt.figure(figsize=figsize)

        csv = data_avd[env]["online"]["DQN"]
        plt_csv(csv, "Online", False, "black")

        for a, algo in enumerate(algos_):
            csv = data_avd[env][mode][algo]
            plt_csv(csv, algo, True, f"C{a+1}")

        _, _, ymin, ymax = plt.axis()
        bottom = max(y_bounds[env][0], ymin)
        top = min(y_bounds[env][1], ymax)
        plt.ylim(bottom=bottom, top=top)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + mode + "." + image_type))
        plt.close()

#############################
#        Comparisons        #
#############################

##################################
# load action-value deviation data
##################################
indir = os.path.join("..", "..", "results", "csv", "action_value_deviation")
outdir = os.path.join("..", "..", "results", "img", "comp_avd")
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data_avd = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    # first hundred invalid, as they are not the correct sma!
    csv = csv[100:]

    if not data_avd.keys() or env not in data_avd.keys():
        data_avd[env] = dict()
    if not data_avd[env].keys() or mode not in data_avd[env].keys():
        data_avd[env][mode] = dict()

    # get reward and use that for obtaining the value deviation and also skip first 100
    csv_ = data[env][mode][algo][100:]
    data_avd[env][mode][algo] = (np.mean(csv, axis=1)[np.argmax(np.mean(csv_, axis=1))],
                                 np.std(csv, axis=1)[np.argmax(np.mean(csv_, axis=1))])


###############
# plot metrics + policy for action value deviation
###############
metrics = {(1,0):"Normalized Return (mean)", (1,1):"Normalized Return (std)", (2,0):"Entropy (mean)",
           (2,1):"Entropy (std)", (4,0):"Episode Length (mean)", (4,1):"Episode Length (std)",
           (5,0):"Unique States per Episode (mean)", (5,1):"Unique States per Episode (std)",
           (6,0):"Uniqueness (mean)", (6,1):"Uniqueness (std)", 7:"Unique States"}

annotations = ["(R)", "(M)", "(ER)", "(N)", "(F)"]
modes = ["random", "mixed", "er", "noisy", "fully"]

for metric in metrics.keys():
    for env in envs:
        plt.figure(figsize=figsize)
        plt.ylabel("Action-Value deviation")
        plt.xlabel(metrics[metric])
        plt.title(env)

        # BC has no value estimate
        algos_ = algos.copy()
        algos_.remove("BC")
        for a, algo in enumerate(algos_):
            x, y, sd = [], [], []
            for mode in modes:
                if metric == 7:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data_avd[env][mode][algo][0])
                sd.append(data_avd[env][mode][algo][1])

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            plt.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{a+1}")
            plt.plot(x, y, "o-", label=algo, color=f"C{a+1}")

        _, _, ymin, ymax = plt.axis()
        bottom = max(y_bounds[env][0], ymin)
        top = min(y_bounds[env][1], ymax)
        plt.ylim(bottom=bottom, top=top)

        xmax, xmin, x_ = 0, 9e9, []
        for m, mode in enumerate(modes):
            if metric == 7:
                x = mm.get_data(env, mode)[metric]
            else:
                x = mm.get_data(env, mode)[metric[0]][metric[1]]
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            x_.append(x)

        # adjust markings if they overlap! do multiple times to be sure
        for _ in range(10):
            adjusted, no_changes = [], True
            for i in range(len(x_)):
                for j in range(len(x_)):
                    if i != j and i not in adjusted and abs(x_[i] - x_[j]) < 0.055 * (xmax - xmin):
                        if x_[i] < x_[j]:
                            x_[i] -= 0.01 * (xmax - xmin)
                            x_[j] += 0.01 * (xmax - xmin)
                        else:
                            x_[i] += 0.01 * (xmax - xmin)
                            x_[j] -= 0.01 * (xmax - xmin)
                        adjusted.append(j)
                        no_changes = False
            if no_changes:
                break

        # position text
        _, _, ymin, ymax = plt.axis()
        for m, x in enumerate(x_):
            plt.text(x, ymin + (ymax - ymin)*0.02, annotations[m], ha="center")

        plt.xlim(right=xmax + (xmax - xmin) * 0.2)

        # Online Policy
        csv = data_avd[env]["online"]["DQN"]
        plt.hlines(y=csv[0], xmin=xmin, xmax=xmax, color="black", linewidths=2, label="Online")

        plt.legend(loc="upper right", fontsize="x-small", markerscale=0.7, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + metrics[metric] + "." + image_type))
        plt.close()

    # plot for modes
    for env in envs:
        plt.figure(figsize=figsize)
        plt.ylabel("Action-Value deviation")
        plt.xlabel("Buffer Type")
        plt.title(env)

        algos_ = algos.copy()
        algos_.remove("BC")
        for a, algo in enumerate(algos_):
            x, y, sd = [], [], []
            for m, mode in enumerate(modes):
                x.append(m)
                y.append(data_avd[env][mode][algo][0])
                sd.append(data_avd[env][mode][algo][1])
            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            plt.fill_between(x, cis[0], cis[1], alpha=0.2, color=F"C{a+1}")
            plt.plot(x, y, "o-", label=algo, color=F"C{a+1}")

        x = []
        for m, mode in enumerate(modes):
            x.append(m)

        _, _, ymin, ymax = plt.axis()
        bottom = max(y_bounds[env][0], ymin)
        top = min(y_bounds[env][1], ymax)
        plt.ylim(bottom=bottom, top=top)

        plt.xlim(right=(len(modes)-1) * 1.2)

        # Online Policy
        csv = data_avd[env]["online"]["DQN"]
        plt.hlines(y=csv[0], xmin=0, xmax=len(modes) - 1, color="black", linewidths=2, label="Online")

        plt.xticks(range(len(modes)), [buffer[m] for m in modes], fontsize="small")

        plt.legend(loc="upper right", fontsize="x-small", markerscale=0.7, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_buffertypes" + "." + image_type))
        plt.close()


##################
# load reward data
##################
indir = os.path.join("..", "..", "results", "csv", "return")
outdir = os.path.join("..", "..", "results", "img", "comp_return")
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    # first hundred invalid, as they are not the correct sma!
    csv = csv[100:]

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or mode not in data[env].keys():
        data[env][mode] = dict()

    # normalize reward
    csv -= random_rewards[envs[env]]
    csv /= (optimal_rewards[envs[env]] - random_rewards[envs[env]])

    data[env][mode][algo] = (np.mean(csv, axis=1).max(), np.std(csv, axis=1)[np.argmax(np.mean(csv, axis=1))])

###############
# plot metrics + policy for reward
###############
metrics = {(1,0):"Normalized Return (mean)", (1,1):"Normalized Return (std)", (2,0):"Entropy (mean)",
           (2,1):"Entropy (std)", (4,0):"Episode Length (mean)", (4,1):"Episode Length (std)",
           (5,0):"Unique States per Episode (mean)", (5,1):"Unique States per Episode (std)",
           (6,0):"Uniqueness (mean)", (6,1):"Uniqueness (std)", 7:"Unique States"}

annotations = ["(R)", "(M)", "(ER)", "(N)", "(F)"]
modes = ["random", "mixed", "er", "noisy", "fully"]

for metric in metrics.keys():
    for env in envs:
        plt.figure(figsize=figsize)
        plt.ylim(bottom=-0.06, top=1.05)
        plt.ylabel("Normalized Return")
        plt.xlabel(metrics[metric])
        plt.title(env)

        for algo in algos:
            x, y, sd = [], [], []
            for mode in modes:
                if metric == 7:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data[env][mode][algo][0])
                sd.append(data[env][mode][algo][1])

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            plt.fill_between(x, cis[0], cis[1], alpha=0.2)
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

        xmax, xmin, x_ = 0, 9e9, []
        for m, mode in enumerate(modes):
            if metric == 7:
                x = mm.get_data(env, mode)[metric]
            else:
                x = mm.get_data(env, mode)[metric[0]][metric[1]]
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            x_.append(x)

        # adjust markings if they overlap! do multiple times to be sure
        for _ in range(10):
            adjusted, no_changes = [], True
            for i in range(len(x_)):
                for j in range(len(x_)):
                    if i != j and i not in adjusted and abs(x_[i] - x_[j]) < 0.055 * (xmax - xmin):
                        if x_[i] < x_[j]:
                            x_[i] -= 0.01 * (xmax - xmin)
                            x_[j] += 0.01 * (xmax - xmin)
                        else:
                            x_[i] += 0.01 * (xmax - xmin)
                            x_[j] -= 0.01 * (xmax - xmin)
                        adjusted.append(j)
                        no_changes = False
            if no_changes:
                break

        for m, x in enumerate(x_):
            plt.text(x, -0.045, annotations[m], ha="center")

        plt.xlim(right=xmax + (xmax - xmin) * 0.2)

        # Online Policy
        csv = data[env]["online"]["DQN"]
        plt.hlines(y=csv[0], xmin=xmin, xmax=xmax, color="black", linewidths=2, label="Online")

        plt.legend(loc="upper right", fontsize="x-small", markerscale=0.7, handlelength=1.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, env + "_" + metrics[metric] + "." + image_type))
        plt.close()

    # plot for modes
    for env in envs:
        plt.figure(figsize=figsize)
        plt.ylim(bottom=-0.06, top=1.05)
        plt.ylabel("Normalized Return")
        plt.xlabel("Buffer Type")
        plt.title(env)

        for algo in algos:
            x, y, sd = [], [], []
            for m, mode in enumerate(modes):
                x.append(m)
                y.append(data[env][mode][algo][0])
                sd.append(data[env][mode][algo][1])
            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            plt.fill_between(x, cis[0], cis[1], alpha=0.2)
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


