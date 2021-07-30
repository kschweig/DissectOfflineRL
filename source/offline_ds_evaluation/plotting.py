import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="Set2")

def plot_histograms(output, returns, episode_length, entropy, action, sparsity):
    bins=30
    type="png"

    fig, axs = plt.subplots(2, 3, figsize=(9,4))
    axs[0, 0].hist(returns, bins=bins, range=(0, 1.01))
    axs[0, 0].set_title('Return')
    axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])

    axs[0, 1].hist(action, bins=(2 * (np.max(action)) + 1), range=(np.min(action) - 0.25, np.max(action) + 0.25))
    axs[0, 1].set_title('Action')
    axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 1].locator_params(axis='x', integer=True)

    axs[0, 2].hist(entropy, bins=bins, range=(0, 1.01))
    axs[0, 2].set_title('Entropy')
    axs[0, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 2].set_xticks([0, 0.25, 0.5, 0.75, 1])


    axs[1, 0].hist(sparsity, bins=bins, range=(0, 1.01))
    axs[1, 0].set_title('Sparsity')
    axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])

    axs[1, 1].hist(episode_length, bins=bins, range=(0, np.max(episode_length)))
    axs[1, 1].set_title('Episode Length')
    axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 1].locator_params(axis='x', integer=True)

    plt.tight_layout()
    plt.savefig(output+"."+type)
    plt.show()


def plot_returns(output, returns, buffer_types, bins=30):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True, sharex=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

    hmin = min([min(rets) for rets in returns])
    hmax = max([max(rets) for rets in returns])

    if hmax > 1. and hmax < 50:
        bins = int(hmax)

    for r, rets in enumerate(returns):
        axs[0].hist(rets, bins=bins, range=(hmin, hmax+1e-7), histtype='step', color=f"C{r}")
        axs[0].set_title('Overview')
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for m in range(1, len(axs)):
        axs[m].hist(returns[m-1], bins=bins, range=(hmin, hmax+1e-7), color=f"C{m-1}")
        axs[m].set_title(buffer_types[m-1])
        axs[m].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.text(0.005, 0.5, "Counts", va='center', rotation='vertical', fontsize="large")
    fig.text(0.52, 0.01, "Return", ha='center', fontsize="large")
    fig.tight_layout(rect=(0.008, 0.022, 1, 1))
    plt.savefig(output)
    plt.close()


def plot_actions(output, actions, buffer_types):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True, sharex=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

    hmin = min([min(acts) for acts in actions]) - 0.25
    hmax = max([max(acts) for acts in actions]) + 0.25

    for r, acts in enumerate(actions):
        axs[0].hist(acts, bins=(2 * (np.max(acts)) + 1), range=(hmin, hmax+1e-7), histtype='step', color=f"C{r}")
        axs[0].set_title('Overview')
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for m in range(1, len(axs)):
        axs[m].hist(actions[m-1], bins=(2 * (np.max(acts)) + 1), range=(hmin, hmax+1e-7), color=f"C{m-1}")
        axs[m].set_title(buffer_types[m-1])
        axs[m].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.text(0.005, 0.5, "Counts", va='center', rotation='vertical', fontsize="large")
    fig.text(0.52, 0.01, "Action", ha='center', fontsize="large")
    fig.tight_layout(rect=(0.008, 0.022, 1, 1))
    plt.savefig(output)
    plt.close()


def plot_entropies(output, entropies, buffer_types, bins=30):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True, sharex=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

    for r, ents in enumerate(entropies):
        axs[0].hist(ents, bins=bins, range=(0, 1.01), histtype='step', color=f"C{r}")
        axs[0].set_title('Overview')
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for m in range(1, len(axs)):
        axs[m].hist(entropies[m-1], bins=bins, range=(0, 1.01), color=f"C{m-1}")
        axs[m].set_title(buffer_types[m-1])
        axs[m].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.text(0.005, 0.5, "Counts", va='center', rotation='vertical', fontsize="large")
    fig.text(0.52, 0.01, "Entropy", ha='center', fontsize="large")
    fig.tight_layout(rect=(0.008, 0.022, 1, 1))
    plt.savefig(output)
    plt.close()


def plot_eplengths(output, ep_lengths, buffer_types, bins=30):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True, sharex=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

    hmin = min([min(epls) for epls in ep_lengths])
    hmax = max([max(epls) for epls in ep_lengths])

    for r, epls in enumerate(ep_lengths):
        axs[0].hist(epls, bins=bins, range=(hmin, hmax+1e-7), histtype='step', color=f"C{r}")
        axs[0].set_title('Overview')
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for m in range(1, len(axs)):
        axs[m].hist(ep_lengths[m-1], bins=bins, range=(hmin, hmax+1e-7), color=f"C{m-1}")
        axs[m].set_title(buffer_types[m-1])
        axs[m].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.text(0.005, 0.5, "Counts", va='center', rotation='vertical', fontsize="large")
    fig.text(0.52, 0.01, "Episode Length", ha='center', fontsize="large")
    fig.tight_layout(rect=(0.008, 0.022, 1, 1))
    plt.savefig(output)
    plt.close()


def plot_sparsities(output, sparsities, buffer_types, bins=30):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True, sharex=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

    for r, sps in enumerate(sparsities):
        axs[0].hist(sps, bins=bins, range=(0, 1.01), histtype='step', color=f"C{r}")
        axs[0].set_title('Overview')
        axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for m in range(1, len(axs)):
        axs[m].hist(sparsities[m-1], bins=bins, range=(0, 1.01), color=f"C{m-1}")
        axs[m].set_title(buffer_types[m-1])
        axs[m].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.text(0.005, 0.5, "Counts", va='center', rotation='vertical', fontsize="large")
    fig.text(0.52, 0.01, "Sparsity", ha='center', fontsize="large")
    fig.tight_layout(rect=(0.008, 0.022, 1, 1))
    plt.savefig(output)
    plt.close()


def plot_states(self, env, buffer, states, _dones, path):
    plt.figure(figsize=(4,3))
    plt.scatter(states[:, 0], states[:, 1])

    dones = []
    for d, done in enumerate(_dones):
        if done:
            dones.append(d + 1)

    plt.title(f"{env} @ {buffer}")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.plot(states[:dones[0], 0], states[:dones[0], 1], "-o", color="black")
    plt.plot(states[dones[len(dones) // 2]:dones[len(dones) // 2 + 1], 0],
             states[dones[len(dones) // 2]:dones[len(dones) // 2 + 1], 1], "-o", color="red")
    plt.plot(states[dones[-2]:dones[-1], 0], states[dones[-2]:dones[-1], 1], "-o", color="blue")
    plt.plot(states[0, 0], states[0, 1], "*", color="black", markersize=12)
    plt.plot(states[dones[len(dones) // 2], 0], states[dones[len(dones) // 2], 1], "*", color="red", markersize=12)
    plt.plot(states[dones[-2], 0], states[dones[-2], 1], marker="*", color="blue", markersize=12)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()

