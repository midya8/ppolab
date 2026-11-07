"""PPO loss (clipped surrogate + value + entropy) with hand-derived gradients,
and GAE(lambda) advantage estimation. Everything checked against finite
differences and brute-force reference implementations in the test suite."""
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Generalized advantage estimation over a flat rollout.

    `dones[i]` is True when transition i ends the episode by true termination
    (no bootstrap past it). Time-limit truncations should NOT be marked done;
    pass the value of the post-truncation state as `next_value` instead.
    Returns (advantages, returns_to_go).
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T = len(rewards)
    adv = np.zeros(T)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = next_value
        else:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


def ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret,
                       clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Scalar PPO loss plus d loss/d logits and d loss/d values.

    All averaging is over the batch; gradient arrays are already divided by B.
    """
    logits = np.asarray(logits, dtype=float)
    values = np.asarray(values, dtype=float)
    actions = np.asarray(actions)
    old_logp = np.asarray(old_logp, dtype=float)
    adv = np.asarray(adv, dtype=float)
    ret = np.asarray(ret, dtype=float)
    B = logits.shape[0]

    logp_all = log_softmax(logits)
    idx = np.arange(B)
    logp = logp_all[idx, actions]
    ratio = np.exp(logp - old_logp)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -np.minimum(surr1, surr2)
    value_loss = 0.5 * (values - ret) ** 2
    probs = np.exp(logp_all)
    entropy = -(probs * logp_all).sum(axis=1)

    loss = policy_loss.mean() + vf_coef * value_loss.mean() - ent_coef * entropy.mean()

    # --- gradients ---
    dlogits = np.zeros_like(logits)
    # policy term: loss_i = -min(surr1, surr2); min picks surr1 where surr1 <= surr2
    via_surr1 = surr1 <= surr2
    dlogp = np.where(via_surr1, -adv, 0.0) * ratio / B
    dlogits[idx, actions] += dlogp
    dlogits -= probs * dlogp[:, None]          # d logp_a / d z_k = delta_ak - p_k
    # value term
    dvalues = vf_coef * (values - ret) / B
    # entropy term: dH/dz_k = -p_k (H + log p_k); loss has -ent_coef * H.mean()
    dlogits += (ent_coef / B) * probs * (entropy[:, None] + logp_all)
    return loss, dlogits, dvalues
