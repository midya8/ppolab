# PPO episode boundaries and replay limits

This is an implementation audit of standard PPO/GAE, not a new algorithm.
The original checkpoint, results, model/loss/optimizer/physics/GAE helper,
18 original tests and images remain unchanged, with hashes in
`src/ppolab/data/original-hashes.json`. The training collector is deliberately
corrected. Historical performance is not evidence of the correction's benefit.

## Collector reset bug

The original `train.rollout` reset at either failure or time-limit truncation,
but recorded only failure in `done_buf`. At an internal timeout, the subsequent
value belonged to the newly reset episode. `compute_gae` used that reset value
and recursively carried its advantage into the previous episode. At a tail
timeout it also evaluated the reset state rather than the state reached.
The existing truncation test covered a tail value supplied by a caller, not
the collector's internal reset behavior.

`boundaries.collect_rollout` retains `next_observations` and `next_values`
before reset, with separate `terminated` and `truncated` flags. Training calls
`compute_gae_transitions`. Define b_t=1-terminated_t and
c_t=1-(terminated_t OR truncated_t):

```
delta[t] = reward[t] + gamma*b[t]*V(actual_post_step_state[t]) - value[t]
advantage[t] = delta[t] + gamma*lambda*c[t]*advantage[t+1]
value_target[t] = advantage[t] + value[t]
```

The unobserved carry beyond the finite rollout is zero, but the final observed
transition still uses its actual successor value unless truly terminated.
The explicit reference sums products of gamma*lambda across nonreset edges.
The full contribution matrix contains the exact zeros across resets.

This convention treats the 500-step cap as an external limit for learning a
continuing balance policy. If time is part of a genuinely finite-horizon task,
a natural ending is termination and the state should include remaining time.
It is not correct to bootstrap every event called a timeout without this task
interpretation. We do not claim the value estimates themselves are correct.

## Compatibility adapter

`train.rollout` still returns seven arrays in the old order. Its returned
rewards now add `gamma*V(actual successor)` at truncation, and its returned
`dones` cut both reset types. Passing these to unchanged `compute_gae` gives
the same corrected result. The adapter's gamma (default 0.99) must match the
GAE caller's gamma. These are augmented rewards, not raw environment rewards.
Use `collect_rollout` for raw rewards and unambiguous flags. Training itself
uses explicit transition GAE, not the adapter. Sampling/RNG order, network,
PPO loss, whole-rollout advantage normalization, Adam and dynamics are retained.
One extra network evaluation stores the actual successor value before reset;
this is not presented as an efficiency improvement.

## GAE JSON audit

```
python -m ppolab.audit gae --input examples/gae.json
```

Exactly: rewards,values,next_values,terminated,truncated,gamma,lam,tail_value.
Each numeric vector has 1..64 aligned entries, absolute value <= 1000. Gamma
and lambda lie in [0,1]; flags are actual JSON booleans and mutually exclusive.
A nonboundary successor value must match the next row's value within 1e-10.
`tail_value` is the original collector's final post-reset value, used only by
the old-bug diagnostic; correct GAE ignores it. At a nonreset tail it normally
matches the actual final successor value. Callers must supply honest states
and values: numeric validation cannot prove the same value function was used.
The NumPy training API accepts up to 100000 transitions under the same bounds.

Four complete captured trajectories use actual original environment physics:
12 steps with a 3-step cap; 32 with a 7-step cap; 64 from original seed 0 untrained
weights with true failures; 64 from the unchanged trained checkpoint at the
original 500-step cap. They include all transitions, actual successor states,
actions/log-probabilities/values and the tail reset value, not selected rows.
Six constructed boundary controls are labeled separately. The old diagnostic
recreates the old reset-value/bootstrap/trace semantics; the second wrong
control treats truncation as termination and loses its bootstrap.

## PPO JSON audit

```
python -m ppolab.audit ppo --input examples/ppo.json
```

Exactly: logits,values,actions,old_logp,advantages,returns,clip_eps,vf_coef,
ent_coef. Logits shape 1..16 x 2..8, absolute logits <= 100, integer action indices,
old action log-probabilities in [-100,0], remaining row values in [-1000,1000].
Epsilon in [0,0.9], value coefficient in [0,2], entropy coefficient in [0,1].
Old action probabilities must be meaningful caller data; they do not identify
a complete old policy or suffice to certify full policy KL.

```
r = exp(log pi_new(action|state) - old_logp)
policy_loss = -min(r*A, clip(r,1-eps,1+eps)*A)
value_loss = 0.5*(value-return)^2
loss = mean(policy_loss + vf_coef*value_loss - ent_coef*entropy)
```

All gradients include 1/B averaging. On equality the original code uses the
unclipped branch. A ratio outside the interval can still have active policy
gradients, depending on advantage sign. Clipped policy gradients do not remove
entropy/value gradients or enforce a hard KL constraint. The 15 constructed
controls cross all combinations of positive/negative/zero advantage and ratios
0.5,0.8,1,1.2,1.5. One complete four-row tiny original ActorCritic batch is a
seeded generated input, not claimed to be an original training minibatch.
All 57 logit/value finite-difference stencils use h=1e-5*max(1,abs(value)).
Four stencils switch branches at a clip kink; both one-sided slopes are
reported, with central-gradient error deliberately null there. A kink has
no unique ordinary derivative. Smooth-stencil normalized errors must be < 1e-8.

JSON input is bounded to 100000 UTF-8 bytes and nesting depth 8. Duplicate keys,
nonfinite numbers including exponent overflow, unknown fields, malformed
shapes and boolean-as-number values are rejected. CLI errors exit 2, print
stderr only and produce no partial JSON. Browser edits clear dependent outputs
and disable exports; arbitrary state simulation rejects invalid bounds.

## Full checkpoint and replay limits

The browser imports every original parameter, not a reduced network. It
implements both tanh hidden layers, both policy logits and the value head
with scalar loops, plus all original Euler equations. Greedy ties choose
left, like NumPy argmax. All 30 initial states come from the full 20-episode
seed 12345 historical final evaluation and 10-episode seed 987 original gate.
All native 14871 transitions are used in independent one-step checks, including
the 373- and 498-step failures. No frames or bad episodes are removed. State input
bounds in the UI are x +/-2.4, angle +/-0.2 rad, both velocities +/-2.

Tests regenerate every original observation, logit, value, action and actual
next state from the unchanged NPZ. Node and actual Chrome evaluate the scalar
network and physics on every original observation; values agree within 1e-11
normalized error and original greedy decisions agree. The reference Node run
observed maximum normalized one-step error 1.0081e-15.

One-step agreement is not trajectory equivalence. Free-running physics amplifies tiny
scalar/NumPy reduction differences. In the checked-in reference run, 377 of
14871 actions differed and maximum absolute state-coordinate difference was
0.59337, while all 30 return lengths happened to agree. Chrome and Node can
also diverge from each other. `scripts/replay_audit.cjs` reports these rather
than loosening a trajectory-equality tolerance or substituting stored frames.
The browser labels paths as new scalar runs and downloads the entire current
path. These counts are a measured platform-specific reference, not guaranteed
across NumPy/BLAS/CPU/V8 versions. A 500-step replay is not a universal solve
claim. A supplied softmax is a policy distribution, not calibrated confidence.

## Verification and install

```
pip install -e '.[dev]'
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 node --test tests/*.cjs
PYTHONPATH=src python scripts/verify_regeneration.py
PYTHONPATH=src node scripts/replay_audit.cjs
bash examples/verify_wheel.sh
```

The boundary-audit stage had 50 Python tests, including 18 unchanged original
tests. The current suite includes these tests. Independent 80-digit Decimal
explicit sums verify all 186 GAE rows. All 26 complete reports contain 12408 numeric
values compared across Python/scalar JS; 57 PPO stencils and original NumPy loss
and derivatives are checked separately. The four Node/actual-Chrome tests added at that stage include
all 14871 original one-step observations, all 30 free-path selectors, every GAE
contribution row, four actual JSON downloads, malformed input clearing,
custom-state errors, play/pause, keyboard selection and 320/390/1280 layouts.
Use Node 22+ with built-in WebSocket and installed Google Chrome. `CHROME_BIN`
overrides its command, and `PYTHON` overrides the test subprocess interpreter.
`PPO_URL` can target the deployed site for the same behavior checks.

Isolated install outside checkout verifies all 8 module origins, all 26 reports,
full original 20-episode evaluation, corrected short-timeout training and both
JSON CLI commands plus the real training CLI. Regeneration uses 1e-10 ordinary
numeric tolerance and 1e-8 finite-difference slope tolerance across runtimes,
checks full structures and restores checked-in files afterward. The scalar audit explicitly uses sequential binary64 reductions on every
supported Python version, rather than Python 3.12's changed built-in sum.
Original network training remains NumPy binary64, not symbolic arithmetic.

A corrected seed 0 full 60-iteration-cap rerun stopped at 12 and reproduced the
historical 493.6 +/- 27.7 final return display. That is one run, not a causal study
or a speed claim. It does not show the bug was exercised in that run, establish
better generalization or justify replacing the original checkpoint.

## Sources and attribution

- Schulman et al., GAE, 2015/2018 version: https://arxiv.org/html/1506.02438v6
  Section 3 derives the discounted residual sum and bias/variance interpretation.
- Pardo et al., Time Limits in Reinforcement Learning, 2018:
  https://ar5iv.labs.arxiv.org/html/1712.00378
  Distinguishes finite-horizon time awareness from partial-episode bootstrap.
- Farama, Gymnasium 1.4 time-limit tutorial:
  https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
  Bootstrap from truncations, not genuine terminations.
- OpenAI Spinning Up PPO:
  https://spinningup.openai.com/en/latest/algorithms/ppo.html
  Sign-dependent clipped surrogate and warning that clipping is not a constraint.
- Costa Huang, 2020 implementation checklist (marked deprecated by its author):
  https://costa.sh/blog-the-32-implementation-details-of-ppo
  Practical loss/normalization/minibatch details; not an ablation study or
  a claim that this project implements every listed optimization.
- Stable Baselines3 issue 1864, 2024:
  https://github.com/DLR-RM/stable-baselines3/issues/1864
  Community explanation of actual terminal-observation reward augmentation.
  The reported SB3 bug was a caller misunderstanding, closed after clarification;
  this is not evidence SB3 has the bug found in this repository.

All sources were read. Formulas are standard and attributed. No framework
model implementation was copied; the original project supplies its own model,
physics and optimizer. Audit/visualization code is project implementation.
