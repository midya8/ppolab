import numpy as np

from ppolab.ppo import compute_gae


def brute_force_gae(rewards, values, dones, next_value, gamma, lam):
    """Reference: explicit discounted sum of future TD residuals, cut at dones."""
    T = len(rewards)
    adv = np.zeros(T)
    for t in range(T):
        gaelam = 0.0
        coeff = 1.0
        for l in range(t, T):
            nextv = next_value if l == T - 1 else values[l + 1]
            nonterminal = 0.0 if dones[l] else 1.0
            delta = rewards[l] + gamma * nextv * nonterminal - values[l]
            gaelam += coeff * delta
            coeff *= gamma * lam * nonterminal
            if dones[l]:
                break
        adv[t] = gaelam
    return adv


def test_matches_brute_force_single_episode():
    rng = np.random.default_rng(0)
    T = 12
    rewards, values = rng.normal(size=T), rng.normal(size=T)
    dones = np.zeros(T, dtype=bool)
    adv, ret = compute_gae(rewards, values, dones, next_value=0.7, gamma=0.9, lam=0.8)
    np.testing.assert_allclose(adv, brute_force_gae(rewards, values, dones, 0.7, 0.9, 0.8), atol=1e-10)
    np.testing.assert_allclose(ret, adv + values, atol=1e-12)


def test_matches_brute_force_with_dones():
    rng = np.random.default_rng(1)
    T = 20
    rewards, values = rng.normal(size=T), rng.normal(size=T)
    dones = rng.random(T) < 0.15
    dones[-1] = True
    adv, _ = compute_gae(rewards, values, dones, next_value=1.3, gamma=0.95, lam=0.9)
    np.testing.assert_allclose(adv, brute_force_gae(rewards, values, dones, 1.3, 0.95, 0.9), atol=1e-10)


def test_done_cuts_credit_assignment():
    # a reward two steps before a done must not leak credit past the done
    rewards = np.array([1.0, 1.0, 5.0, 0.0, 0.0])
    values = np.zeros(5)
    dones = np.array([False, False, True, False, False])
    adv, ret = compute_gae(rewards, values, dones, next_value=0.0, gamma=0.9, lam=1.0)
    assert adv[2] == 5.0
    assert adv[1] == 1.0 + 0.9 * 5.0
    assert adv[3] == 0.0  # fresh episode, no leakage from the previous one


def test_truncation_bootstraps_like_an_unbroken_episode():
    values = np.array([1.0, 1.0, 1.0])
    rewards = np.array([1.0, 1.0, 1.0])
    dones = np.array([False, False, False])
    adv_boot, _ = compute_gae(rewards, values, dones, next_value=2.0, gamma=0.9, lam=1.0)
    # equivalent to extending the episode with reward r = v(next) - v(next) = 0... but
    # bootstrap adds gamma * next_value to the last delta, so compare directly:
    np.testing.assert_allclose(adv_boot[-1], 1.0 + 0.9 * 2.0 - 1.0)
    np.testing.assert_allclose(adv_boot[1], 0.9 * (adv_boot[2] + 1.0))
    np.testing.assert_allclose(adv_boot[0], 0.9 * (adv_boot[1] + 1.0))
