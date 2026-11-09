"""Transition-wise GAE. Bootstrap at truncation, but never recurse across a reset."""
import numpy as np
from .ppo import log_softmax, softmax


def compute_gae_transitions(rewards, values, next_values, terminated, truncated,
                            gamma=0.99, lam=0.95):
    from .audit import validate_gae
    if any(not isinstance(t, (bool, np.bool_)) for seq in (terminated,truncated) for t in seq):
        raise ValueError("boundary flags must be boolean")
    x = validate_gae(dict(rewards=list(rewards), values=list(values),
        next_values=list(next_values), terminated=[bool(t) for t in terminated],
        truncated=[bool(t) for t in truncated], gamma=gamma, lam=lam, tail_value=0.0), limit=100000)
    r, v, nv = [np.asarray(x[k], dtype=float) for k in ("rewards", "values", "next_values")]
    term, trunc = [np.asarray(x[k], dtype=bool) for k in ("terminated", "truncated")]
    delta = r + gamma * nv * (~term) - v
    adv = np.zeros(len(r)); carry = 0.0
    for t in range(len(r) - 1, -1, -1):
        carry = delta[t] + gamma * lam * (not (term[t] or trunc[t])) * carry
        adv[t] = carry
    return adv, adv + v


def collect_rollout(net, env, horizon, rng):
    """Keep actual post-step states/values BEFORE reset, with two boundary flags.

    Starts with env.reset(), like the original rollout. Does not concatenate
    recursive credit across reset. RNG/action sampling and physics are unchanged.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 100000:
        raise ValueError("horizon must be an integer in 1..100000")
    keys = ("observations", "next_observations", "actions", "old_logp", "rewards",
            "values", "next_values", "terminated", "truncated")
    rows = {k: [] for k in keys}; obs = env.reset()
    for _ in range(horizon):
        logits, value, _ = net.forward(obs[None, :])
        lp = log_softmax(logits)[0]
        action = int(rng.choice(len(lp), p=softmax(logits)[0]))
        next_obs, reward, term, trunc = env.step(action)
        _, next_value, _ = net.forward(next_obs[None, :])
        for k, v in zip(keys, (obs.copy(), next_obs.copy(), action, float(lp[action]),
                               reward, float(value[0]), float(next_value[0]), term, trunc)):
            rows[k].append(v)
        obs = env.reset() if term or trunc else next_obs
    out = {k: np.asarray(v) for k, v in rows.items()}
    _, tail, _ = net.forward(obs[None, :])
    out["tail_value"] = float(tail[0])
    return out
