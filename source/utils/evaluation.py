import numpy as np
import warnings


def evaluate(env, agent, writer, all_rewards, over_episodes=100):
    done, ep_reward, values, values_std, action_values, entropies = False, 0, [], [], [], []
    state = env.reset()

    while not done:
        action, value, entropy = agent.policy(state, eval=True)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
        values.append(value.numpy().mean())
        values_std.append(value.numpy().std())
        action_values.append(value.max().item())
        entropies.append(entropy)

    all_rewards.append(ep_reward)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        writer.add_scalar("eval/Reward", ep_reward, len(all_rewards))
        writer.add_scalar("eval/Reward (SMA)", np.mean(all_rewards[-over_episodes:]), len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (mean)", np.nanmean(action_values), len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (std)", np.nanstd(action_values), len(all_rewards))
        writer.add_scalar("eval/Values", np.nanmean(values), len(all_rewards))
        writer.add_scalar("eval/Action-Values std", np.nanmean(values_std), len(all_rewards))
        writer.add_scalar("eval/Entropy", np.nanmean(entropies), len(all_rewards))
    return all_rewards


def entropy(values):
    probs = values.detach().cpu().numpy()
    # if entropy degrades
    if np.min(probs) < 1e-5:
        return 0
    return -np.sum(probs * np.log(probs))
