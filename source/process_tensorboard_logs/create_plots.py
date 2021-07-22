import os
import glob
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)


image_type = "png"
figsize = (12, 6)
figsize_comp = (12, 7)

# metric manager
experiments = ["ex4", "ex5", "ex6"]

mm = MetricsManager(0)


for ex in experiments:
    paths = glob.glob(os.path.join("..", "..", "data", ex, "metrics*.pkl"))
    for path in paths:
        with open(path, "rb") as f:
            m = pickle.load(f)
        mm.data.update(m.data)

# static stuff

envs = {'CartPole-v1': 0, 'MountainCar-v0': 1, "MiniGrid-LavaGapS7-v0": 2, "MiniGrid-Dynamic-Obstacles-8x8-v0": 3,
        'Breakout-MinAtar-v0': 4, "Space_invaders-MinAtar-v0": 5}

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]

buffer = {"random": "Random Policy", "mixed": "Mixed Policy", "er": "Experience Replay",
          "noisy": "Noisy Policy", "fully": "Final Policy"}

y_bounds = {'CartPole-v1': (-15, 15), "MiniGrid-LavaGapS7-v0":(-0.5, 1.3), 'MountainCar-v0': (-50, 100),
            "MiniGrid-Dynamic-Obstacles-8x8-v0":(-1, 1), 'Breakout-MinAtar-v0': (-5, 25), "Space_invaders-MinAtar-v0": (-5, 25)}

metrics = {(0,0):"Return", (0,1):"Return (std)",
           1:"Unique States", 2:"Unique State-Action Pairs",
           (3,0):"Entropy", (3,1):"Entropy (std)",
           (4,0):"Sparsity", (4,1): "Sparsity (std)",
           (5,0):"Episode Length", (5,1):"Episode Length (std)",
           }

annotations = ["(R)", "(M)", "(E)", "(N)", "(F)"]


def plt_csv(ax, csv, algo, mode, ylims=None, set_title=True, color=None, set_label=True):
    est = np.mean(csv, axis=1)
    sd = np.std(csv, axis=1)
    cis = (est - sd, est + sd)

    ax.fill_between(np.arange(0, len(est) * 100, 100), cis[0], cis[1], alpha=0.2, color=color)
    ax.plot(np.arange(0, len(est) * 100, 100), est, label=(algo if set_label else None), color=color)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if set_title:
        ax.set_title(buffer[mode])
    if ylims != None:
        ax.set_ylim(bottom=ylims[0], top=ylims[1])


####################################
#       Usual Return plots         #
####################################

mark = "return"

# titles
y_label = "Return"
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

    data[env][mode][algo] = csv

for env in data.keys():

    f, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    axs = [item for sublist in axs for item in sublist]

    for m, mode in enumerate(data[env].keys()):

        if mode == "online":
            continue

        ids = list(buffer.keys())
        ax = axs[ids.index(mode)]

        norm = mm.get_data(env, mode)[0][0]
        ax.axhline(y=norm, color="black", linestyle="dotted",
                   linewidth=2, label=("Behav." if m==0 else None))
                   
        csv = data[env]["online"]["DQN"]
        ax.axhline(y=csv.max(), color="black", linewidth=2)
        plt_csv(ax, csv, "Online", mode, color="black", set_label=m==0)

        for a, algo in enumerate(algos):
            csv = data[env][mode][algo]
            plt_csv(ax, csv, algo, mode, color=f"C{a}", set_label=m==0)

    for ax in axs[m:]:
        f.delaxes(ax)

    f.legend(loc="lower right", bbox_to_anchor=(0.89, 0.06))
    f.tight_layout(rect=(0.008, 0, 1, 1))
    f.text(0.52, 0.02, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, env + "." + image_type))
    plt.close()


##################################
#    Action-Value Deviations     #
##################################

mark = "avd"

# titles
y_label = "Action-Value Deviation"
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

    f, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    axs = [item for sublist in axs for item in sublist]

    for m, mode in enumerate(data_avd[env].keys()):

        if mode == "online":
            continue

        ids = list(buffer.keys())
        ax = axs[ids.index(mode)]

        ax.axhline(y=0, color="black", linewidth=2, linestyle="dotted", label=("True" if m==0 else None))

        csv = data_avd[env]["online"]["DQN"]

        bottom, top = list(), list()
        plt_csv(ax, csv, "Online", mode, color="black", set_label=m==0)

        for a, algo in enumerate(algos_):
            csv = data_avd[env][mode][algo]
            plt_csv(ax, csv, algo, mode, color=f"C{a+1}", set_label=m==0)

        ax.set_ylim(bottom=y_bounds[env][0], top=y_bounds[env][1])



    for ax in axs[m:]:
        f.delaxes(ax)

    #axs[2].xaxis.set_tick_params(labelbottom=True)

    f.legend(loc="lower right", bbox_to_anchor=(0.89, 0.09))
    f.tight_layout(rect=(0.008, 0, 1, 1))
    f.text(0.52, 0.02, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, env + "." + image_type))
    plt.close()



#############################
#        Comparisons        #
#############################

##################################
# load action-value deviation data
##################################
indir = os.path.join("..", "..", "results", "csv", "avd")
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
modes = list(buffer.keys())

# titles
y_label = "Action-Value Deviation"
x_label = "Buffer Types"


for metric in metrics.keys():

    f, axs = plt.subplots(2, 3, figsize=figsize)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env[:-3])

        # BC has no value estimate
        algos_ = algos.copy()
        algos_.remove("BC")
        for a, algo in enumerate(algos_):
            x, y, sd = [], [], []
            for mode in modes:
                if metric == 1 or metric == 2:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data_avd[env][mode][algo][0])
                sd.append(data_avd[env][mode][algo][1])

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{a+1}")
            ax.plot(x, y, "o-", label=(algo if e == 1 else None), color=f"C{a+1}")

        ax.set_ylim(bottom=y_bounds[env][0], top=y_bounds[env][1])

        xmax, xmin, x_ = 0, 9e9, []
        for m, mode in enumerate(modes):
            if metric == 1 or metric == 2:
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
                    if i != j and i not in adjusted and abs(x_[i] - x_[j]) < 0.08 * (xmax - xmin):
                        delta = 0.08 * (xmax - xmin) - abs(x_[i] - x_[j])
                        if x_[i] < x_[j]:
                            x_[i] -= delta / 2
                            x_[j] += delta / 2
                        else:
                            x_[i] += delta / 2
                            x_[j] -= delta / 2
                        adjusted.append(j)
                        no_changes = False
            if no_changes:
                break

        # position text
        _, _, ymin, ymax = ax.axis()
        ax.set_ylim(ymin - (ymax - ymin) * 0.08, ymax)
        for m, x in enumerate(x_):
            ax.text(x, ymin - (ymax - ymin) * 0.05, annotations[m], ha="center")

        ax.axhline(y=0, color="black", linestyle="dotted", label=("True" if e == 0 else None))

        # Online Policy
        csv = data_avd[env]["online"]["DQN"]
        ax.axhline(y=csv[0], color="black", label=("Online" if e==0 else None))


    f.legend(loc="upper center", ncol=len(algos_) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
    f.text(0.52, 0.01, metrics[metric], ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, metrics[metric] + "." + image_type))
    plt.close()

# plot for modes
f, axs = plt.subplots(2, 3, figsize=figsize, sharex=True)
axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env[:-3])

    ax.axhline(y=0, color="black", linestyle="dotted", label=("True" if e == 0 else None))
    # Online Policy
    csv = data_avd[env]["online"]["DQN"]
    ax.axhline(y=csv[0], color="black", label=("Online" if e == 0 else None))

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
        ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=F"C{a+1}")
        ax.plot(x, y, "o-", label=(algo if e == 0 else None), color=F"C{a+1}")

    x = []
    for m, mode in enumerate(modes):
        x.append(m)

    ax.set_ylim(bottom=y_bounds[env][0], top=y_bounds[env][1])

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(algos_) + 2, fontsize="small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.95), w_pad=-0.5)
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "buffertypes." + image_type))
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

    data[env][mode][algo] = (np.mean(csv, axis=1).max(), np.std(csv, axis=1)[np.argmax(np.mean(csv, axis=1))])

###############
# plot metrics + policy for reward
###############

# titles
y_label = "Return"
x_label = "Buffer Types"


for metric in metrics.keys():

    f, axs = plt.subplots(2, 3, figsize=figsize)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env[:-3])

        for a, algo in enumerate(algos):
            x, y, sd = [], [], []
            for mode in modes:
                if metric == 1 or metric == 2:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data[env][mode][algo][0])
                sd.append(data[env][mode][algo][1])

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{a}")
            ax.plot(x, y, "o-", label=(algo if e == 1 else None), color=f"C{a}")

        x, y = [], []
        for mode in modes:
            if metric == 1 or metric == 2:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
            y.append(mm.get_data(env, mode)[0][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e==0 else None), color="black")

        xmax, xmin, x_ = 0, 9e9, []
        for m, mode in enumerate(modes):
            if metric == 1 or metric == 2:
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
                    if i != j and i not in adjusted and abs(x_[i] - x_[j]) < 0.08 * (xmax - xmin):
                        delta = 0.08 * (xmax - xmin) - abs(x_[i] - x_[j])
                        if x_[i] < x_[j]:
                            x_[i] -= delta / 2
                            x_[j] += delta / 2
                        else:
                            x_[i] += delta / 2
                            x_[j] -= delta / 2
                        adjusted.append(j)
                        no_changes = False
            if no_changes:
                break

        # position text
        _, _, ymin, ymax = ax.axis()
        ax.set_ylim(ymin - (ymax - ymin) * 0.08, ymax)
        for m, x in enumerate(x_):
            ax.text(x, ymin - (ymax - ymin)*0.05, annotations[m], ha="center")

        # Online Policy
        csv = data[env]["online"]["DQN"]
        ax.axhline(y=csv[0], color="black", label=("Online" if e==0 else None))


    f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
    f.text(0.52, 0.01, metrics[metric], ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, metrics[metric] + "." + image_type))
    plt.close()

# plot for modes
f, axs = plt.subplots(2, 3, figsize=figsize, sharex=True)
axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env[:-3])

    x, y = list(range(len(buffer))), []
    for mode in modes:
        y.append(mm.get_data(env, mode)[0][0])
    x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

    ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e == 0 else None), color="black")

    # Online Policy
    csv = data[env]["online"]["DQN"]
    ax.axhline(y=csv[0], color="black", label=("Online" if e == 0 else None))

    for a, algo in enumerate(algos):

        x, y, sd = [], [], []
        for m, mode in enumerate(modes):
            x.append(m)
            y.append(data[env][mode][algo][0])
            sd.append(data[env][mode][algo][1])
        x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

        cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
        ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=F"C{a}")
        ax.plot(x, y, "o-", label=(algo if e == 0 else None), color=F"C{a}")

    x = []
    for m, mode in enumerate(modes):
        x.append(m)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "buffertypes." + image_type))
plt.close()
