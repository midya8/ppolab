import numpy as np

from ppolab.optim import Adam


def test_adam_minimizes_quadratic():
    w = np.array([5.0, -3.0])
    opt = Adam({"w": w}, lr=0.1)
    for _ in range(500):
        g = 2 * opt.params["w"]  # grad of w^2
        opt.step({"w": g})
    assert np.all(np.abs(opt.params["w"]) < 1e-3)
