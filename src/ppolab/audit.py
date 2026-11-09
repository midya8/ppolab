"""Strict, bounded scalar GAE and PPO-loss ledgers, independent of NumPy."""
import json
import math

def _total(values):
    out = 0.0
    for v in values: out += v
    return out


def read_json(text):
    if len(text.encode("utf-8")) > 100000:
        raise ValueError("input exceeds 100000 bytes")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate key: " + key)
            out[key] = value
        return out
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (RecursionError, json.JSONDecodeError) as e:
        raise ValueError("invalid JSON") from e
    def depth(x, n=0):
        if n > 8: raise ValueError("input nesting exceeds 8")
        if isinstance(x,float) and not math.isfinite(x):raise ValueError("nonfinite JSON")
        if isinstance(x, dict):
            for v in x.values(): depth(v,n+1)
        if isinstance(x, list):
            for v in x: depth(v,n+1)
    depth(obj)
    return obj


def number(x, name, lo=-1000, hi=1000):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(name + " is not a bounded finite number")
    return float(x)


def exact(x, keys):
    if not isinstance(x, dict) or set(x) != set(keys.split()):
        raise ValueError("expected exactly: " + keys)


def vector(x, name, n=None, lo=-1000, hi=1000):
    if not isinstance(x, list) or not x or (n is not None and len(x) != n): raise ValueError(name + " shape")
    return [number(v,name,lo,hi) for v in x]


def validate_gae(x, limit=64):
    exact(x,"rewards values next_values terminated truncated gamma lam tail_value")
    r=vector(x["rewards"],"rewards")
    if len(r)>limit: raise ValueError("too many transitions")
    n=len(r)
    for key in ("values", "next_values"): vector(x[key],key,n)
    for key in ("terminated", "truncated"):
        if not isinstance(x[key],list) or len(x[key])!=n or any(type(t) is not bool for t in x[key]): raise ValueError(key+" requires boolean flags")
    if any(a and b for a,b in zip(x["terminated"],x["truncated"])): raise ValueError("termination and truncation must be exclusive")
    number(x["gamma"],"gamma",0,1);number(x["lam"],"lam",0,1);number(x["tail_value"],"tail_value")
    for t in range(n-1):
        if not(x["terminated"][t] or x["truncated"][t]) and abs(x["next_values"][t]-x["values"][t+1])>1e-10: raise ValueError("nonboundary successor value must match next row")
    return x


def gae_audit(x):
    validate_gae(x); n=len(x["rewards"]);g=x["gamma"];l=x["lam"]
    delta=[r+g*nv*(not term)-v for r,v,nv,term in zip(x["rewards"],x["values"],x["next_values"],x["terminated"])]
    carry=0.; adv=[0.]*n
    for t in reversed(range(n)):
        carry=delta[t]+g*l*(not(x["terminated"][t] or x["truncated"][t]))*carry;adv[t]=carry
    contributions=[]
    for t in range(n):
        row=[0.]*n;weight=1.
        for j in range(t,n):
            row[j]=weight*delta[j]
            if x["terminated"][j] or x["truncated"][j]: break
            weight*=g*l
        contributions.append(row)
    # Reproduce the old reset-value bootstrap plus cross-reset recursion.
    old=[0.]*n;cut=[0.]*n;a=b=0.
    for t in reversed(range(n)):
        nv=x["values"][t+1] if t+1<n else x["tail_value"]
        a=x["rewards"][t]+g*nv*(not x["terminated"][t])-x["values"][t]+g*l*(not x["terminated"][t])*a
        old[t]=a
        done=x["terminated"][t] or x["truncated"][t]
        b=x["rewards"][t]+g*x["next_values"][t]*(not done)-x["values"][t]+g*l*(not done)*b;cut[t]=b
    rows=[dict(t=t,reward=x["rewards"][t],value=x["values"][t],next_value=x["next_values"][t],terminated=x["terminated"][t],truncated=x["truncated"][t],bootstrap_mask=int(not x["terminated"][t]),trace_mask=int(not(x["terminated"][t] or x["truncated"][t])),delta=delta[t],advantage=adv[t],target=adv[t]+x["values"][t],explicit_sum=_total(contributions[t]),old_reset_advantage=old[t],wrong_terminal_advantage=cut[t]) for t in range(n)]
    return dict(rows=rows,contributions=contributions,max_sum_error=max(abs(adv[t]-_total(contributions[t])) for t in range(n)),max_old_error=max(abs(adv[t]-old[t]) for t in range(n)),max_terminal_error=max(abs(adv[t]-cut[t]) for t in range(n)))


def validate_ppo(x):
    exact(x,"logits values actions old_logp advantages returns clip_eps vf_coef ent_coef")
    z=x["logits"]
    if not isinstance(z,list) or not 1<=len(z)<=16 or not isinstance(z[0],list) or not 2<=len(z[0])<=8: raise ValueError("logits shape 1..16 x 2..8")
    n=len(z);k=len(z[0])
    for row in z: vector(row,"logits",k,-100,100)
    for key in ("values","advantages","returns"):vector(x[key],key,n)
    vector(x["old_logp"],"old_logp",n,-100,0)
    if not isinstance(x["actions"],list) or len(x["actions"])!=n or any(isinstance(a,bool) or not isinstance(a,(int,float)) or not float(a).is_integer() or not 0<=a<k for a in x["actions"]):raise ValueError("actions shape/index")
    number(x["clip_eps"],"clip_eps",0,0.9);number(x["vf_coef"],"vf_coef",0,2);number(x["ent_coef"],"ent_coef",0,1)
    return x


def ppo_audit(x, finite=True):
    validate_ppo(x);n=len(x["logits"]);eps=x["clip_eps"];rows=[];total=0.
    for i,z in enumerate(x["logits"]):
        m=max(z);lse=math.log(_total(math.exp(v-m) for v in z));lp=[v-m-lse for v in z];p=[math.exp(v) for v in lp]
        a=int(x["actions"][i]);A=x["advantages"][i];ratio=math.exp(lp[a]-x["old_logp"][i]);clipped=min(1+eps,max(1-eps,ratio));s1=ratio*A;s2=clipped*A;via=s1<=s2
        policy=-min(s1,s2);value=.5*(x["values"][i]-x["returns"][i])**2;H=-_total(q*v for q,v in zip(p,lp));loss=(policy+x["vf_coef"]*value-x["ent_coef"]*H)/n
        dlp=-A*ratio/n if via else 0.;dz=[dlp*((1 if j==a else 0)-q)+x["ent_coef"]/n*q*(H+lp[j]) for j,q in enumerate(p)];dv=x["vf_coef"]*(x["values"][i]-x["returns"][i])/n
        rows.append(dict(t=i,probabilities=p,log_probabilities=lp,action=a,advantage=A,ratio=ratio,clipped_ratio=clipped,surrogate_unclipped=s1,surrogate_clipped=s2,via_unclipped=via,ratio_outside=(ratio<1-eps or ratio>1+eps),policy_loss=policy,value_loss=value,entropy=H,mean_loss_contribution=loss,dlogits=dz,dvalue=dv));total+=loss
    out=dict(loss=total,rows=rows)
    if finite:
        import copy
        checks=[]
        for i,row in enumerate(rows):
            for key,j,analytic in [("logits",j,v) for j,v in enumerate(row["dlogits"])]+[("values",None,row["dvalue"])]:
                val=x[key][i] if j is None else x[key][i][j];h=1e-5*max(1,abs(val));pair=[];branches=[]
                for sign in (-1,1):
                    y=copy.deepcopy(x)
                    if j is None:y[key][i]+=sign*h
                    else:y[key][i][j]+=sign*h
                    r=ppo_audit(y,finite=False);pair.append(r["loss"]);branches.append(r["rows"][i]["via_unclipped"])
                left=(total-pair[0])/h;right=(pair[1]-total)/h;smooth=branches[0]==branches[1]
                checks.append(dict(row=i,field=key,coordinate=j,analytic=analytic,left=left,right=right,central=(pair[1]-pair[0])/(2*h),smooth=smooth,normalized_error=(abs((pair[1]-pair[0])/(2*h)-analytic)/max(1,abs(analytic)) if smooth else None)))
        out["checks"]=checks;out["max_smooth_error"]=max([c["normalized_error"] for c in checks if c["smooth"]]+[0])
    return out


def main():
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument("kind",choices=["gae","ppo"]);p.add_argument("--input",required=True);a=p.parse_args()
    try:
        if a.input=="-":text=sys.stdin.buffer.read(100001).decode("utf-8")
        else:
            with open(a.input,"rb") as f:text=f.read(100001).decode("utf-8")
        x=read_json(text);r=(gae_audit if a.kind=="gae" else ppo_audit)(x);print(json.dumps(r,allow_nan=False))
    except (ValueError,TypeError,OverflowError,OSError,UnicodeError) as e:print(str(e),file=sys.stderr);sys.exit(2)

if __name__=="__main__":main()
