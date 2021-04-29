import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="Set2")


def plot_histograms(output, reward, episode_length, intersections, free_path_length, entropy, action,
                    sparsity, state_sim, start_sim):
    bins=30
    type="png"

    fig, axs = plt.subplots(3, 3, figsize=(9,6))
    axs[0, 0].hist(reward, bins=bins)
    axs[0, 0].set_title('Reward')
    axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 1].hist(action, bins=(2 * np.max(action) + 1))
    axs[0, 1].set_title('Action')
    axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0, 2].hist(entropy, bins=bins, range=(0, 1))
    axs[0, 2].set_title('Entropy')
    axs[0, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axs[1, 0].hist(episode_length, bins=bins, range=(0, np.max(episode_length)))
    axs[1, 0].set_title('Episode Length')
    axs[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 1].hist(intersections, bins=bins, range=(0, np.max(intersections)))
    axs[1, 1].set_title('Intersections')
    axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 2].hist(free_path_length, bins=bins, range=(0, np.max(free_path_length)))
    axs[1, 2].set_title('Free Path Length')
    axs[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axs[2, 0].hist(sparsity, bins=bins, range=(0, 1))
    axs[2, 0].set_title('Sparsity')
    axs[2, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[2, 1].hist(state_sim, bins=bins, range=(0, 1))
    axs[2, 1].set_title('State Similarity')
    axs[2, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[2, 2].hist(start_sim, bins=bins, range=(0, 1))
    axs[2, 2].set_title('Start Similarity')
    axs[2, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output+"."+type)
    plt.show()
