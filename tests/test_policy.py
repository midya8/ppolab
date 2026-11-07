import os

import numpy as np
import pytest

from ppolab.cartpole import CartPole
from ppolab.nets import ActorCritic

POLICY = os.path.join(os.path.dirname(__file__), "..", "assets", "policy.npz")


@pytest.mark.skipif(not os.path.exists(POLICY), reason="trained checkpoint not present")
def test_trained_policy_solves_cartpole():
    net = ActorCritic.load(POLICY)
    env = CartPole(seed=987)
    returns = []
    for _ in range(10):
        env.reset()
        total, done = 0.0, False
        while not done:
            logits, _, _ = net.forward(env.state[None, :])
            _, _, term, trunc = env.step(int(np.argmax(logits[0])))
            total += 1.0
            done = term or trunc
        returns.append(total)
    assert np.mean(returns) >= 400.0, f"mean return {np.mean(returns)}"
