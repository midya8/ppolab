"""Emit all original per-step policy/physics values for independent JS comparison."""
import json,sys
from pathlib import Path
from ppolab.nets import ActorCritic
from ppolab.cartpole import CartPole
import numpy as np
p=Path(__file__).resolve().parents[1]
model=json.loads((p/'src/ppolab/data/model.json').read_text());net=ActorCritic.load(p/'assets/policy.npz');episodes=[]
for e in model['episodes']:
 env=CartPole(seed=0);env.state=np.array(e['initial']);env.steps=0;rows=[]
 for t in range(500):
  obs=env.state.copy();z,v,_=net.forward(obs);a=int(np.argmax(z[0]));n,r,term,trunc=env.step(a);rows.append(dict(t=t,observation=obs.tolist(),logits=z[0].tolist(),value=float(v[0]),action=a,next_observation=n.tolist(),reward=r,terminated=term,truncated=trunc))
  if term or trunc:break
 episodes.append(rows)
print(json.dumps(episodes,separators=(',',':')))
