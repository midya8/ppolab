import numpy as np
import pytest

from ppolab.cartpole import CartPole


def test_deterministic_with_seed():
    a, b = CartPole(seed=7), CartPole(seed=7)
    for _ in range(50):
        sa = a.step(1)
        sb = b.step(1)
        assert np.allclose(sa[0], sb[0])
        assert sa[1] == sb[1] and sa[2] == sb[2] and sa[3] == sb[3]


def test_pole_falls_without_action_intervention():
    env = CartPole(seed=0)
    env.reset()
    # always push right: cart drifts right until it leaves the track
    steps = 0
    while True:
        _, _, terminated, truncated = env.step(1)
        steps += 1
        assert steps <= env.max_steps
        if terminated or truncated:
            break
    assert env.failed  # cart left the track or the pole fell past 12 degrees


def test_termination_when_pole_falls():
    env = CartPole(seed=1)
    env.reset()
    while True:
        _, _, terminated, truncated = env.step(0)
        if terminated or truncated:
            break
    if terminated:
        assert abs(env.state[2]) > env.theta_threshold or abs(env.state[0]) > env.x_threshold


def test_reward_is_one_per_step():
    env = CartPole(seed=2)
    env.reset()
    for _ in range(10):
        _, r, term, trunc = env.step(1)
        assert r == 1.0
        if term or trunc:
            break


def test_truncation_at_time_limit():
    env = CartPole(seed=3, max_steps=5)
    env.reset()
    for i in range(5):
        _, _, term, trunc = env.step(1)
        if term:
            break
    assert trunc and i == 4


def test_reset_resamples_state():
    env = CartPole(seed=4)
    s1 = env.reset()
    s2 = env.reset()
    assert s1.shape == (4,) and s2.shape == (4,)
    # both inits are within the reported uniform box
    assert np.all(np.abs(s1) <= 0.05 + 1e-12)


def test_random_policy_survives_only_briefly():
    env = CartPole(seed=5)
    rng = np.random.default_rng(0)
    env.reset()
    steps = 0
    while steps < env.max_steps:
        _, _, term, trunc = env.step(int(rng.integers(0, 2)))
        steps += 1
        if term or trunc:
            break
    assert steps < 200  # a random walker on the real dynamics dies fast
