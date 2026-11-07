"""CartPole physics implemented from first principles. No gym dependency."""
import numpy as np


class CartPole:
    """Classic cart-pole balancing task, Euler-integrated.

    State: [x, x_dot, theta, theta_dot]. Actions: 0 pushes left, 1 pushes right.
    Reward is 1 per surviving step. Episode ends (terminated) when the pole
    falls past 12 degrees or the cart leaves the track, and is truncated at
    `max_steps` (time-limit truncation, still bootstrappable).
    """

    def __init__(self, seed=0, max_steps=500):
        self.g = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # half-length of the pole
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.dt = 0.02
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = None
        self.last_termination = None
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        self.last_termination = None
        return self.state.copy()

    def _dynamics(self, s, action):
        x, x_dot, theta, theta_dot = s
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        return np.array([x, x_dot, theta, theta_dot])

    def step(self, action):
        assert action in (0, 1)
        self.state = self._dynamics(self.state, action)
        self.steps += 1
        x, x_dot, theta, theta_dot = self.state
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        self.last_termination = "terminated" if terminated else ("truncated" if truncated else None)
        return self.state.copy(), 1.0, terminated, truncated

    @property
    def failed(self):
        x, _, theta, _ = self.state
        return abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
