"""A tiny actor-critic MLP in pure NumPy with hand-written backprop."""
import numpy as np


class ActorCritic:
    """Trunk: 4 -> hidden -> hidden with tanh. Two heads: 2 action logits, 1 value."""

    def __init__(self, obs_dim=4, act_dim=2, hidden=64, seed=0, logit_scale=0.01):
        rng = np.random.default_rng(seed)

        def xavier(fin, fout):
            return rng.normal(0.0, np.sqrt(2.0 / (fin + fout)), size=(fin, fout))

        self.p = {
            "W1": xavier(obs_dim, hidden), "b1": np.zeros(hidden),
            "W2": xavier(hidden, hidden), "b2": np.zeros(hidden),
            "Wp": rng.normal(0.0, logit_scale, size=(hidden, act_dim)), "bp": np.zeros(act_dim),
            "Wv": xavier(hidden, 1), "bv": np.zeros(1),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.rng = rng

    def forward(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        h1 = np.tanh(x @ self.p["W1"] + self.p["b1"])
        h2 = np.tanh(h1 @ self.p["W2"] + self.p["b2"])
        logits = h2 @ self.p["Wp"] + self.p["bp"]
        value = (h2 @ self.p["Wv"] + self.p["bv"])[:, 0]
        return logits, value, (x, h1, h2)

    def backward(self, cache, dlogits, dvalue):
        """Accumulate d loss/d params given output gradients (already mean-scaled)."""
        x, h1, h2 = cache
        g = self.grads
        g["Wp"][:] += h2.T @ dlogits
        g["bp"][:] += dlogits.sum(axis=0)
        dv = np.asarray(dvalue, dtype=float)[:, None]
        g["Wv"][:] += h2.T @ dv
        g["bv"][:] += dv.sum(axis=0)
        dz2 = dlogits @ self.p["Wp"].T + dv @ self.p["Wv"].T
        dz2 = dz2 * (1.0 - h2 ** 2)
        g["W2"][:] += h1.T @ dz2
        g["b2"][:] += dz2.sum(axis=0)
        dz1 = (dz2 @ self.p["W2"].T) * (1.0 - h1 ** 2)
        g["W1"][:] += x.T @ dz1
        g["b1"][:] += dz1.sum(axis=0)

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][:] = 0.0

    def state_dict(self):
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, sd):
        for k in self.p:
            self.p[k] = np.asarray(sd[k], dtype=float).copy()

    def save(self, path):
        np.savez(path, **self.state_dict())

    @classmethod
    def load(cls, path, **kwargs):
        data = np.load(path)
        net = cls(**kwargs)
        net.load_state_dict({k: data[k] for k in data.files})
        return net
