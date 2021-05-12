import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="Set2")


def plot_histograms(output, reward, episode_length, unique_states_episode, entropy, action, sparsity):
    bins=30
    type="png"

    fig, axs = plt.subplots(2, 3, figsize=(9,4))
    axs[0, 0].hist(reward, bins=bins, range=(0, 1))
    axs[0, 0].set_title('Reward')
    axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])

    axs[0, 1].hist(action, bins=(2 * (np.max(action)) + 1), range=(np.min(action) - 0.25, np.max(action) + 0.25))
    axs[0, 1].set_title('Action')
    axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 1].locator_params(axis='x', integer=True)

    axs[0, 2].hist(entropy, bins=bins, range=(0, 1))
    axs[0, 2].set_title('Entropy')
    axs[0, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 2].set_xticks([0, 0.25, 0.5, 0.75, 1])


    axs[1, 0].hist(sparsity, bins=bins, range=(0, 1))
    axs[1, 0].set_title('Sparsity')
    axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])

    axs[1, 1].hist(episode_length, bins=bins, range=(0, np.max(episode_length)))
    axs[1, 1].set_title('Episode Length')
    axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 1].locator_params(axis='x', integer=True)

    axs[1, 2].hist(unique_states_episode, bins=bins, range=(0, 1))
    axs[1, 2].set_title('Uniqueness')
    axs[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 2].locator_params(axis='x', integer=True)
    axs[1, 2].set_xticks([0, 0.25, 0.5, 0.75, 1])

    plt.tight_layout()
    plt.savefig(output+"."+type)
    plt.show()
