import numpy as np
import warnings
import torch.nn.functional as F


def evaluate(env, agent, writer, all_rewards, over_episodes=100):
    done, ep_reward, values, entropies = False, 0, [], []
    state = env.reset()

    while not done:
        action, value, entropy = agent.policy(state, eval=True)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
        values.append(value)
        entropies.append(entropy)

    all_rewards.append(ep_reward)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        writer.add_scalar("eval/Reward", ep_reward, len(all_rewards))
        writer.add_scalar("eval/Reward (SMA)", np.mean(all_rewards[-over_episodes:]), len(all_rewards))
        writer.add_scalar("eval/Values", np.nanmean(values), len(all_rewards))
        writer.add_scalar("eval/Entropy", np.nanmean(entropies), len(all_rewards))

    return all_rewards


def entropy(values):
    probs = F.softmax(values, dim=1).detach().cpu().numpy()
    return -np.sum(probs * np.log(probs))

