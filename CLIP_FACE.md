# PPO-Clip: exact optimal-policy faces

This extension audits the **single-state full old-policy expectation** of the standard clipped PPO surrogate. It does not replace the original trainer, checkpoint, CartPole evaluation, boundary inspector or GAE implementation. Those retain their original numerical semantics.

## The question

Clipping a term in an objective is different from restricting the policy's probability ratio. What are *all* policies attaining the best clipped objective? How close or far can those equally optimal policies be from the old policy? Which exact witnesses prove those claims?

[Interactive exact face inspector](https://midya8.github.io/ppolab/clip.html).

The implementation is a small original audit of established PPO and separable concave optimization, not a new PPO algorithm, trust-region method, benchmark or training result. The mathematical counterexample is developed for inspection, not sampled from CartPole.

## Declared problem

For 2–6 actions, old probabilities `p_i > 0` sum exactly to one. Supplied rational advantages `A_i` may be positive, negative or zero. They are never silently centered or normalized. A supplied advantage table need not be a true advantage function; the report shows `sum p_i A_i` explicitly. Epsilon satisfies `0 <= e < 1`.

On the closed probability simplex `q_i >= 0`, `sum q_i = 1`, maximize

```
F(q) = sum_i p_i min((q_i/p_i) A_i,
                     clip(q_i/p_i, 1-e, 1+e) A_i).
```

The factor `p_i` is the expectation weight, not an optional multiplier. This is not the empirical average of one observed action per state in an arbitrary PPO minibatch. It is the exact all-action expectation at one supplied state. An independent test constructs a repeated-action batch with empirical frequencies equal to p and compares its score with the **unchanged original NumPy PPO loss**, with entropy/value coefficients zero.

Each contribution is concave piecewise linear:

- `A_i > 0`: `f_i(q) = A_i min(q, (1+e)p_i)`.
- `A_i < 0`: `f_i(q) = A_i max(q, (1-e)p_i)`.
- `A_i = 0`: `f_i(q) = 0`.

The sign determines the effective breakpoint. Lower clipping does not cap the positive-advantage term, and upper clipping does not cap the negative-advantage term.

The report solves this same objective twice:

1. `simplex`: only probability nonnegativity and normalization.
2. `ratio_box`: additionally `(1-e)p_i <= q_i <= (1+e)p_i`, with upper endpoints capped at one.

The second domain is a real hard ratio constraint, **not** an alternative interpretation of the first objective. The exact objective gap is reported, including zero when the constraint does not reduce the optimum.

## Exact dual and complete face

For any multiplier lambda and coordinate domain `[l_i,u_i]`, let

```
h_i(lambda) = max_{q in [l_i,u_i]} [f_i(q) - lambda q].
```

Every feasible normalized q satisfies `F(q) <= lambda + sum h_i`. The producer scans candidate multipliers among `{0,A_1,...,A_n}`. These are all possible piece slopes. At a dual optimum a multiplier can be chosen at a slope: between successive slopes the coordinate maximizers are unchanged, and an endpoint either already allows mass one or supplies a tied segment allowing it. Each coordinate maximum needs only its domain endpoints and its in-domain clipping breakpoint.

For a multiplier whose coordinate argmax intervals `[L_i,U_i]` satisfy `sum L_i <= 1 <= sum U_i`, normalized mass can be assigned within those intervals. This attains the bound, proving global optimality exactly.

It also describes **all** optimizers, not just one: each coordinate supporting gap is nonnegative, and their sum is the total dual gap. A feasible policy attains the bound if and only if every coordinate gap vanishes. Consequently the complete optimal face is

```
L_i <= q_i <= U_i for every i, and sum q_i = 1.
```

The effective attaining marginal ranges are tightened to

```
low_i  = max(L_i, 1 - sum_{j != i} U_j)
high_i = min(U_i, 1 - sum_{j != i} L_j).
```

Each endpoint is attainable, but the ranges are not independent choices. The sum-to-one condition always remains.

A box-simplex vertex fixes at least n−1 coordinates at endpoints. The producer enumerates each possible free coordinate and every choice of the other endpoint coordinates, retains feasible normalized policies, deduplicates exact tuples and sorts them in rational lexicographic order. This is the complete geometric vertex set, not every piecewise-cell knot or a sample of successful policies. Equal endpoints and lower-dimensional faces are supported.

## Independent replay

`clipverify.py` does not import or call `clipface.py`. For each domain it enumerates *all piecewise cells' vertices*: fix n−1 coordinates at every domain endpoint and every applicable breakpoint, solve the remaining coordinate by normalization, and evaluate the original direct min expression. A linear function on each compact cell attains its maximum at a cell vertex. Thus the exhaustive maximum is an independent global reference.

From all winning cell vertices the verifier reconstructs effective coordinate ranges, enumerates their box-simplex vertices, verifies each is globally optimal, then checks the supporting-line dual certificate. This combination proves the whole box face, rather than assuming that marginal ranges alone prove joint feasibility. It independently checks every row, interval, vertex, objective, TV witness, support flag and positive-family leaf. Unknown/missing fields and altered leaves fail. It accepts a mathematically valid dual multiplier, not a claim that data or authorship is authenticated. A coherent new problem and certificate can verify as that *new declared problem*.

## Exact total variation

`TV(p,q) = (1/2) sum |p_i-q_i|`.

For the optimal box face, clamp p into its effective coordinate ranges. Restore any missing/excess mass inside the bounds. This gives an attaining nearest policy. Independently, every feasible q must pay at least

```
max(sum_i max(low_i-p_i,0), sum_i max(p_i-high_i,0)).
```

The clamped-and-restored witness attains this lower bound. TV is convex, so its maximum over this compact polytope is attained at a vertex; the complete vertex atlas supplies the farthest witness. Ties select the first rational-lexicographic vertex. Neither the nearest nor farthest witness is presented as the unique preferred PPO update.

## Support and old-to-new KL

No floating logarithm is treated as exact arithmetic. The report uses the support argument for `KL(p || q)` only, with this orientation named explicitly.

Since every p_i is positive, a zero coordinate of any optimal policy gives infinite old-to-new KL on the closed simplex. A full-support optimal policy exists if and only if every effective coordinate maximum is positive. The average of all face vertices is then a positive optimal witness.

If both a zero-coordinate optimal vertex z and a positive optimal centroid c exist, the entire family

```
q(t) = (1-t) z + t c,  0 < t <= 1
```

is positive and optimal by convexity of the face. For a zero coordinate of z, `q_i(t)=t c_i`, so `p_i log(p_i/q_i(t))` diverges as t decreases to zero. Other coordinate contributions have finite lower bounds. Thus old-to-new KL is unbounded even **among strictly positive exact optimizers**. Three rational members are included, but the unboundedness argument concerns the whole family, not a finite log table.

If some action has effective maximum zero, no positive exact optimum exists. The report distinguishes this empty full-support set from an unbounded positive-optimum family. Finite softmax logits cannot attain a zero probability; a closed-simplex optimum is not automatically an attainable finite-network parameter update.

## A centered control with a strict domain gap

For `p=(1/10,9/10)`, `A=(9,-1)`, `e=1/5`, the old-weighted advantage mean is zero.

- The actual clipped-objective optimum is `9/25`.
- Every optimizer has `q_0 in [7/25,1]` and `q_1=1-q_0`.
- TV ranges from `9/50` to `9/10`, with both endpoints attained.
- A hard ratio box forces `q_0 <= 3/25` and has optimum `1/5`.
- Adding the box loses `4/25` of objective.
- The simplex face includes `(1,0)` and a positive optimal centroid. It therefore has the positive optimal family with unbounded `KL(p || q)` described above.

This exhibits a difference in the **optimization problems**, not proof that the original CartPole SGD follows this family or that a particular production PPO implementation fails.

## APIs and strict JSON

Python API:

```python
from ppolab.clipface import audit
from ppolab.clipverify import verify
report = audit(['1/10','9/10'], [9,-1], '1/5')
verify(report)
```

CLI input has exactly these fields:

```json
{"version":1,"old":["1/10","9/10"],"advantages":[9,-1],"epsilon":"1/5"}
```

```bash
python -m ppolab.clipface input.json > report.json
python -m ppolab.clipverify report.json
```

Integers, `Fraction` values (Python API), exact fraction strings and finite decimal strings are accepted. Floats, booleans, exponent notation, underscores and surrounding whitespace in rational strings are rejected. String length is at most 100; each input numerator/denominator is at most 128 bits. The browser requires JSON integer numbers to be within JavaScript's safe-integer range; use strings for larger integers. The browser uses the existing duplicate-key-rejecting, 100KB/8-level JSON reader. Native input is capped at 100KB; duplicate/unknown fields, wrong version and malformed input exit 2 without JSON output. The independent report CLI caps certificates at 2MB. Report rational strings are canonical reduced fractions or integers. Derived rational outputs can exceed the input bit bound.

The report top level contains exactly: `version`, normalized exact `old`, `advantages`, `epsilon`, `old_mean_advantage`, `simplex`, `ratio_box`, `hard_box_objective_gap`.

Each face contains:

- `domain.lower/upper`; `multiplier`; `rows`, with every domain/breakpoint knot's q, clipped term, adjusted term, coordinate support and argmax interval; `objective`.
- Effective `ranges`; complete sorted `vertices`; aligned `vertex_tv`; `nearest/minimum_tv`; `farthest/maximum_tv`; `centroid`.
- `full_support_optimum_exists`, `forced_zero`, `old_to_new_kl_infinite_on_closed_face`, `zero_witness`, `full_support_kl_unbounded`, and `positive_optimal_approach` members with t, policy and objective.

These are finite exact witnesses. KL flags are support statements, not decimal KL estimates. The browser exports the complete input/certificate, which the native independent verifier can replay. Editing input immediately hides results, clears stale evidence and disables export until valid recomputation.

## Reproduction

The ten original arithmetic fixtures include centered asymmetric and symmetric cases, zero advantages, no positive exact optimum, identical slopes, epsilon zero, unequal three-action terms, six actions, a rare action and all-negative supplied advantages. `scripts/generate_clip_fixtures.py` regenerates both the packaged JSON and browser loader. Neither archive replaces the earlier model or 26 boundary/PPO reports.

Tests include exhaustive ternary advantages for 2, 3 and 4 actions at three epsilon values (351 complete reports), independent piecewise-cell replay, a denominator-24 three-action grid, six-action and tiny-probability controls, permutations/positive scaling, every-leaf tampering and the original NumPy loss expectation bridge. The browser's independently written BigInt engine compares all complete reports, not rounded scores. Actual Chrome visits every fixture/domain/policy selector, checks exact rendered table cells and responsive layouts, performs trusted input invalidation and restore, downloads real JSON Blobs, and invokes native regeneration/verification on those downloaded bytes.

```bash
PYTHONPATH=src python -m pytest -q
node --test tests/*.cjs
bash examples/verify_wheel.sh
bash examples/verify_clip_wheel.sh
```

The original installed gate remains unchanged; the extension gate checks its own installed module origins, complete packaged reports, strict CLI and certificate replay outside checkout. CI and live deployment are separate gates, not assumed from local tests.

## Limits and attribution

No sampled advantage estimation, unseen-state distribution, multi-state shared parameters, value head, entropy bonus, GAE, optimizer dynamics, learning-rate choice, monotonic return improvement or production scaling is inferred. Supplied arithmetic controls are not independent training benchmarks. Exact rational probability geometry is different from numerical softmax evaluation. The original trained artifacts and historical measured returns stay unchanged.

Sources read for this extension:

1. Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347v2), Sections 2–5, especially Equations 7 and 9. Standard surrogate, sign-dependent clipping and separate entropy/value terms.
2. [OpenAI Spinning Up PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html). Explicitly distinguishes clipping from a constraint and names separate KL early stopping.
3. Wang, He and Tan (2020), [Truly Proximal Policy Optimization](https://proceedings.mlr.press/v115/wang20b.html). Existing work on ratio/trust-region limitations; its proposed algorithm and experiments are not implemented here.
4. Engstrom et al. (2020), [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729). Objective versus optimizer/code-level effects; no reproduction of its training results claimed.
5. Sun et al. (2022), [You May Not Need Ratio Clipping in PPO](https://arxiv.org/html/2202.00079). Existing ratio/TV and early-stopping analysis; ESPO is not implemented here.
6. Markowitz and Staley (2023), [Clipped-Objective Policy Gradients for Pessimistic Policy Optimization](https://arxiv.org/abs/2311.05846), Section 3.2. PPO objective attribution, not implementation of COPG.

This source list is attribution, not endorsement. The exact finite optimizer, supporting witnesses, counterexample controls, independent replay and UI are this project's engineering contribution.
