import numpy as np
import warnings


def evaluate(env, agent, writer, all_rewards, all_deviations_mean, all_deviations_std, over_episodes=100):
    done, ep_reward, values, values_std, action_values, entropies, qval_delta = False, [], [], [], [], [], []
    state = env.reset()

    while not done:
        action, value, entropy = agent.policy(state, eval=True)
        state, reward, done, _ = env.step(action)
        ep_reward.append(reward)
        values.append(value.numpy().mean())
        values_std.append(value.numpy().std())
        if len(value.view(-1)) > action:
            action_values.append(value.view(-1)[action].item())
        else:
            action_values.append(np.nan)
        entropies.append(entropy)

    # calculate target discounted reward
    cum_reward, cr = np.zeros_like(ep_reward), 0
    for i in reversed(range(len(ep_reward))):
        cr = cr + ep_reward[i]
        cum_reward[i] = cr
        cr *= agent.discount

    for i, qval in enumerate(action_values):
        # compare action value with real outcome
        qval_delta.append(qval - cum_reward[i])

    all_rewards.append(sum(ep_reward))
    all_deviations_mean.append(np.mean(qval_delta))
    all_deviations_std.append(np.std(qval_delta))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        writer.add_scalar("eval/Reward", sum(ep_reward), len(all_rewards))
        writer.add_scalar("eval/Reward (SMA)", np.nanmean(all_rewards[-over_episodes:]), len(all_rewards))
        writer.add_scalar("eval/Action-Value deviation (mean)", np.nanmean(qval_delta), len(all_rewards))
        writer.add_scalar("eval/Action-Value deviation (mean) (SMA)", np.nanmean(all_deviations_mean[-over_episodes:]),
                          len(all_rewards))
        writer.add_scalar("eval/Action-Value deviation (std)", np.nanstd(qval_delta), len(all_rewards))
        writer.add_scalar("eval/Action-Value deviation (std) (SMA)", np.nanmean(all_deviations_std[-over_episodes:]),
                          len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (mean)", np.nanmean(action_values), len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (std)", np.nanstd(action_values), len(all_rewards))
        writer.add_scalar("eval/Values", np.nanmean(values), len(all_rewards))
        writer.add_scalar("eval/Action-Values std", np.nanmean(values_std), len(all_rewards))
        writer.add_scalar("eval/Entropy", np.nanmean(entropies), len(all_rewards))

    return all_rewards, all_deviations_mean, all_deviations_std


def entropy(values):
    probs = values.detach().cpu().numpy()
    # if entropy degrades
    if np.min(probs) < 1e-5:
        return 0
    return -np.sum(probs * np.log(probs))
