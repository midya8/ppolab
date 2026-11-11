"""Exact finite categorical PPO-Clip objective faces, not a PPO trainer.

All probabilities and advantages describe one supplied state. No entropy,
value loss, parameter sharing, sample noise or dynamics are modeled here.
"""
from fractions import Fraction as F
from itertools import product
import json
import re


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, F)):
        raise ValueError('use integer or rational/decimal string, not float/bool')
    if isinstance(x, str) and (not x or len(x) > 100 or not re.fullmatch(r'[+-]?(?:[0-9]+/[0-9]+|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)', x)):
        raise ValueError('rational string length must be 1..100')
    try:
        q = F(x)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError('invalid rational') from e
    if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > 128:
        raise ValueError('rational input exceeds 128 bits')
    return q


def problem(old, advantages, epsilon):
    if not isinstance(old, (list, tuple)) or not 2 <= len(old) <= 6:
        raise ValueError('need 2..6 old probabilities')
    if not isinstance(advantages, (list, tuple)) or len(advantages) != len(old):
        raise ValueError('advantages must align with probabilities')
    p, a, e = list(map(rational, old)), list(map(rational, advantages)), rational(epsilon)
    if any(x <= 0 for x in p) or sum(p) != 1:
        raise ValueError('old probabilities must be positive and sum exactly to one')
    if not 0 <= e < 1:
        raise ValueError('epsilon must be in [0,1)')
    return p, a, e


def term(p, a, e, q):
    ratio = q / p
    clipped = min(1 + e, max(1 - e, ratio))
    return p * min(ratio * a, clipped * a)


def _vertices(lo, hi):
    out = set()
    n = len(lo)
    for free in range(n):
        others = [i for i in range(n) if i != free]
        for bits in product((0, 1), repeat=n - 1):
            q = list(lo)
            for i, bit in zip(others, bits):
                q[i] = hi[i] if bit else lo[i]
            q[free] = 1 - sum(q[i] for i in others)
            if lo[free] <= q[free] <= hi[free]:
                out.add(tuple(q))
    return sorted(out)


def _face(p, a, e, lower, upper):
    knots = []
    for pi, ai, l, u in zip(p, a, lower, upper):
        t = pi * (1 + e if ai > 0 else 1 - e)
        knots.append(sorted({l, u} | ({t} if l < t < u and ai else set())))
    # Concave separable resource allocation. A dual multiplier can always be
    # chosen at an available piece slope (including zero).
    for multiplier in sorted(set(a + [F(0)])):
        rows, lo, hi = [], [], []
        for pi, ai, ks in zip(p, a, knots):
            values = [term(pi, ai, e, q) - multiplier * q for q in ks]
            best = max(values)
            winning = [q for q, v in zip(ks, values) if v == best]
            lo.append(min(winning)); hi.append(max(winning))
            rows.append({'knots': [{'q': q, 'term': term(pi, ai, e, q),
                                    'adjusted': v} for q, v in zip(ks, values)],
                         'support': best, 'argmax': [min(winning), max(winning)]})
        if sum(lo) <= 1 <= sum(hi):
            break
    else:
        raise AssertionError('no dual support for feasible box simplex')
    # Intersect the independent argmax intervals with sum(q)=1. These are
    # effective marginal ranges, not independently selectable probabilities.
    low = [max(l, 1 - sum(hi[j] for j in range(len(p)) if j != i)) for i, l in enumerate(lo)]
    high = [min(u, 1 - sum(lo[j] for j in range(len(p)) if j != i)) for i, u in enumerate(hi)]
    vertices = _vertices(low, high)
    assert vertices
    objective = multiplier + sum(row['support'] for row in rows)
    # Nearest old policy in TV: clamp then restore mass inside the optimal box.
    nearest = [min(u, max(l, pi)) for pi, l, u in zip(p, low, high)]
    residual = 1 - sum(nearest)
    for i in range(len(p)):
        change = min(residual, high[i] - nearest[i]) if residual >= 0 else max(residual, low[i] - nearest[i])
        nearest[i] += change; residual -= change
    assert residual == 0
    tv = lambda q: sum(abs(qi - pi) for qi, pi in zip(q, p)) / 2
    farthest = max(vertices, key=tv)
    centroid = [sum(q[i] for q in vertices) / len(vertices) for i in range(len(p))]
    zero_vertex = next((q for q in vertices if 0 in q), None)
    full_support = all(x > 0 for x in centroid)
    approach = []
    if zero_vertex is not None and full_support:
        for t in [F(1, 2), F(1, 16), F(1, 256)]:
            q = [(1 - t) * z + t * c for z, c in zip(zero_vertex, centroid)]
            approach.append({'t': t, 'policy': q,
                             'objective': sum(term(pi, ai, e, qi) for pi, ai, qi in zip(p, a, q))})
    return {'domain': {'lower': lower, 'upper': upper},
            'multiplier': multiplier, 'rows': rows, 'objective': objective,
            'ranges': list(map(list, zip(low, high))), 'vertices': [list(q) for q in vertices],
            'vertex_tv': [tv(q) for q in vertices], 'nearest': nearest,
            'minimum_tv': tv(nearest), 'farthest': list(farthest), 'maximum_tv': tv(farthest),
            'centroid': centroid, 'full_support_optimum_exists': full_support,
            'forced_zero': [i for i, u in enumerate(high) if u == 0],
            'old_to_new_kl_infinite_on_closed_face': zero_vertex is not None,
            'zero_witness': list(zero_vertex) if zero_vertex is not None else None,
            'full_support_kl_unbounded': zero_vertex is not None and full_support,
            'positive_optimal_approach': approach}


def audit(old, advantages, epsilon):
    p, a, e = problem(old, advantages, epsilon)
    unrestricted = _face(p, a, e, [F(0)] * len(p), [F(1)] * len(p))
    hard = _face(p, a, e, [(1 - e) * x for x in p], [min(F(1), (1 + e) * x) for x in p])
    return encode({'version': 1, 'old': p, 'advantages': a, 'epsilon': e,
                   'old_mean_advantage': sum(x * y for x, y in zip(p, a)),
                   'simplex': unrestricted, 'ratio_box': hard,
                   'hard_box_objective_gap': unrestricted['objective'] - hard['objective']})


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def _pairs(items):
    out = {}
    for k, v in items:
        if k in out:
            raise ValueError('duplicate JSON field')
        out[k] = v
    return out


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    args = parser.parse_args()
    try:
        with open(args.input, 'rb') as f:
            data = f.read(100001)
        if len(data) > 100000:
            raise ValueError('input exceeds 100KB')
        value = json.loads(data, object_pairs_hook=_pairs)
        if not isinstance(value, dict) or set(value) != {'version', 'old', 'advantages', 'epsilon'} or type(value['version']) is not int or value['version'] != 1:
            raise ValueError('expected version 1, old, advantages, epsilon only')
        print(json.dumps(audit(value['old'], value['advantages'], value['epsilon']), sort_keys=True, separators=(',', ':')))
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
