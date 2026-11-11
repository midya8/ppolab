"""Independent exhaustive clip-face certificate replay.

Does not import the optimizer. Enumerates every simplex/hyperplane cell
vertex before checking global optimum, coordinate ranges and face vertices.
The report authenticates its declared exact problem, not data or authorship.
"""
from fractions import Fraction as F
from itertools import combinations, product
import json


def _q(value):
    if not isinstance(value, str) or len(value) > 10000:
        raise ValueError('report rational must be a canonical string')
    q = F(value)
    if str(q) != value:
        raise ValueError('noncanonical rational')
    return q


def _enc(x):
    if isinstance(x, F): return str(x)
    if isinstance(x, list): return [_enc(v) for v in x]
    if isinstance(x, dict): return {k: _enc(v) for k, v in x.items()}
    return x


def _term(pi, ai, ep, qi):
    raw = qi * ai
    edge = pi * (1 + ep if ai >= 0 else 1 - ep) * ai
    return min(raw, edge)


def _points(bounds):
    """Intersection of sum=1 with n-1 independently fixed hyperplanes."""
    n = len(bounds)
    found = set()
    for fixed in combinations(range(n), n - 1):
        free = next(i for i in range(n) if i not in fixed)
        for values in product(*(bounds[i] for i in fixed)):
            q = [F(0)] * n
            for i, value in zip(fixed, values): q[i] = value
            q[free] = 1 - sum(values)
            if all(min(bounds[i]) <= q[i] <= max(bounds[i]) for i in range(n)):
                found.add(tuple(q))
    return sorted(found)


def verify(report):
    """Return True, or raise ValueError for a failed/unknown/missing leaf."""
    try:
        if not isinstance(report, dict) or set(report) != {'version', 'old', 'advantages', 'epsilon', 'old_mean_advantage', 'simplex', 'ratio_box', 'hard_box_objective_gap'} or type(report['version']) is not int or report['version'] != 1:
            raise ValueError('report schema')
        p = [_q(x) for x in report['old']]; a = [_q(x) for x in report['advantages']]; e = _q(report['epsilon'])
        if not 2 <= len(p) <= 6 or len(a) != len(p) or min(p) <= 0 or sum(p) != 1 or not 0 <= e < 1:
            raise ValueError('declared problem')
        if any(max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > 128 for q in p+a+[e]):
            raise ValueError('declared input bit bound')
        expected = {'version': 1, 'old': p, 'advantages': a, 'epsilon': e,
                    'old_mean_advantage': sum(x*y for x,y in zip(p,a))}
        n = len(p)
        for name in ['simplex', 'ratio_box']:
            face = report[name]
            if name == 'simplex': lower, upper = [F(0)]*n, [F(1)]*n
            else: lower, upper = [x*(1-e) for x in p], [min(F(1),x*(1+e)) for x in p]
            knots = []
            for i in range(n):
                t = p[i]*(1+e if a[i] > 0 else 1-e)
                knots.append(sorted(set([lower[i],upper[i]] + ([t] if a[i] and lower[i]<t<upper[i] else []))))
            candidates = _points(knots)
            score = lambda q: sum(_term(x,y,e,z) for x,y,z in zip(p,a,q))
            opt = max(map(score,candidates))
            winners = [q for q in candidates if score(q)==opt]
            lo = [min(q[i] for q in winners) for i in range(n)]
            hi = [max(q[i] for q in winners) for i in range(n)]
            vertices = _points([sorted(set([l,u])) for l,u in zip(lo,hi)])
            if any(score(q)!=opt for q in vertices):
                raise ValueError('non-box optimal face')
            # Validate supporting-line dual evidence independently at all knots.
            lam = _q(face['multiplier']); rows=[]
            for i,ks in enumerate(knots):
                vals=[_term(p[i],a[i],e,q)-lam*q for q in ks]; best=max(vals)
                selected=[q for q,v in zip(ks,vals) if v==best]
                rows.append({'knots':[{'q':q,'term':_term(p[i],a[i],e,q),'adjusted':v} for q,v in zip(ks,vals)],'support':best,'argmax':[min(selected),max(selected)]})
            if lam+sum(r['support'] for r in rows)!=opt:
                raise ValueError('dual gap')
            tv=lambda q:sum(abs(qi-pi) for qi,pi in zip(q,p))/2
            near=[min(hi[i],max(lo[i],p[i])) for i in range(n)]
            missing=1-sum(near)
            for i in range(n):
                if missing>0: change=min(missing,hi[i]-near[i])
                else: change=-min(-missing,near[i]-lo[i])
                near[i]+=change;missing-=change
            independent_min=max(sum(max(F(0),l-pi) for l,pi in zip(lo,p)),sum(max(F(0),pi-u) for u,pi in zip(hi,p)))
            if missing or tv(near)!=independent_min or score(near)!=opt:
                raise ValueError('minimum TV witness')
            max_tv=max(map(tv,vertices));far=next(q for q in vertices if tv(q)==max_tv)
            center=[sum(q[i] for q in vertices)/len(vertices) for i in range(n)]
            zero=next((q for q in vertices if 0 in q),None);positive=min(center)>0
            approach=[]
            if zero is not None and positive:
                for t in [F(1,2),F(1,16),F(1,256)]:
                    q=[(1-t)*z+t*c for z,c in zip(zero,center)]
                    if score(q)!=opt or min(q)<=0:raise ValueError('positive optimal approach')
                    approach.append({'t':t,'policy':q,'objective':score(q)})
            expected[name]={'domain':{'lower':lower,'upper':upper},'multiplier':lam,'rows':rows,'objective':opt,'ranges':[[l,u] for l,u in zip(lo,hi)],'vertices':[list(q) for q in vertices],'vertex_tv':[tv(q) for q in vertices],'nearest':near,'minimum_tv':tv(near),'farthest':list(far),'maximum_tv':max_tv,'centroid':center,'full_support_optimum_exists':positive,'forced_zero':[i for i,u in enumerate(hi) if not u],'old_to_new_kl_infinite_on_closed_face':zero is not None,'zero_witness':list(zero) if zero is not None else None,'full_support_kl_unbounded':zero is not None and positive,'positive_optimal_approach':approach}
        expected['hard_box_objective_gap']=expected['simplex']['objective']-expected['ratio_box']['objective']
        # JSON comparison keeps bool distinct from integer and catches every leaf.
        if json.dumps(_enc(expected),sort_keys=True,separators=(',',':')) != json.dumps(report,sort_keys=True,separators=(',',':')):
            raise ValueError('certificate leaf mismatch')
    except (KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError) as exc:
        raise ValueError('invalid certificate structure') from exc
    return True


def main():
    import argparse,sys
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('report');args=parser.parse_args()
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON field')
            d[k]=v
        return d
    try:
        with open(args.report,'rb') as f:data=f.read(2000001)
        if len(data)>2000000:raise ValueError('report exceeds 2MB')
        verify(json.loads(data,object_pairs_hook=pairs));print('verified declared clip-face certificate')
    except (ValueError,OSError,RecursionError) as exc:
        print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
