# ppolab

Proximal Policy Optimization, from scratch, in pure NumPy. No PyTorch, no JAX, no gym: the clipped surrogate objective, GAE(lambda), Adam, and the CartPole environment itself are all implemented from first principles, with every gradient checked against finite differences and every estimator checked against a brute-force reference.

## What's inside

- `src/ppolab/cartpole.py`: the classic cart-pole task written from the equations of motion (Euler integration, same constants as the standard benchmark). Termination is a real pole failure; hitting the 500-step time limit is a truncation that still bootstraps, and the two are kept apart because mixing them is a classic silent bug in RL code.
- `src/ppolab/nets.py`: a shared-trunk actor-critic MLP (4 -> 64 -> 64) with hand-written backprop through both heads.
- `src/ppolab/ppo.py`: the PPO loss (clipped surrogate + value MSE + entropy bonus) with derived analytic gradients, and GAE(lambda) advantage estimation.
- `src/ppolab/optim.py`: Adam.
- `src/ppolab/train.py`: the full loop: rollouts, advantage normalization, 10 epochs of minibatch updates, greedy evaluation, early stop on solve.

## Honest results

A single run (seed 0, 60-iteration cap, early stop on solve) reached mean greedy eval return **499.8 ± 0.6 over 10 episodes at iteration 12** and finished with **493.6 ± 27.7 over 20 episodes** (max is 500). Training took about 4 seconds of CPU time. Nothing here is tuned to flatter itself: the final 20-episode number is reported as measured, including the couple of episodes where the still-imperfect policy dropped the pole late.

![learning curve](assets/learning_curve.png)

The trained checkpoint ships in `assets/policy.npz`, and the test suite loads it and requires it to hold the pole for a mean of at least 400 steps, so the repo's headline claim is enforced by CI, not asserted in prose.

![demo](assets/cartpole_demo.gif)

## Verification, not vibes

The test suite (18 tests, all passing) does the work prose can't:

- **Finite-difference gradient checks** on the full PPO loss (per-logit, per-value) and on every parameter of the actor-critic network end to end.
- **Brute-force GAE reference**: advantages match an explicit discounted sum of TD residuals, including rollouts cut by terminations mid-way.
- **Credit-assignment isolation**: reward before a `done` must not leak past it, and time-limit truncation must bootstrap exactly like an unbroken episode.
- **Clip semantics**: the objective caps at `(1+eps)*A` when the ratio runs away with positive advantage, the gradient is zero on the clipped branch, and with zero advantage the entropy term provably prefers the uniform policy.
- **Environment checks**: seeded determinism, real physics (constant right-pushing force ends the episode through the state boundary), and a random policy dying in under 200 steps.

## Run it

```bash
pip install numpy matplotlib pillow pytest
PYTHONPATH=src python -m pytest -q           # verification incl. the trained checkpoint
PYTHONPATH=src python -m ppolab.train --iters 60 --out assets   # retrain (~4s, CPU)
```

## Why it's interesting

Most from-scratch PPO write-ups stop at "it works". This repo makes the parts that usually stay implicit explicit and testable: the difference between termination and time-limit truncation in bootstrapping, where the clipped objective's gradient actually flows (and where it deliberately doesn't), and what GAE(lambda) does to credit assignment across episode boundaries. Every one of those behaviors has a test that would catch the standard implementation mistake.

## License

MIT
