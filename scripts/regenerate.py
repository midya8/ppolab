"""Rebuild all public inputs/outputs from unchanged original checkpoint and physics."""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
from ppolab.audit import gae_audit,ppo_audit
from ppolab.boundaries import collect_rollout,compute_gae_transitions
from ppolab.cartpole import CartPole
from ppolab.nets import ActorCritic
from ppolab.ppo import ppo_loss_and_grads,compute_gae,log_softmax
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'src/ppolab/data'
def conv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def write(name,x):
 (out/name).write_text(json.dumps(x,default=conv,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
net=ActorCritic.load(ROOT/'assets/policy.npz');fixtures=[]
for name,horizon,max_steps,seed,trained in [('real/truncated-every-3',12,3,3,True),('real/truncated-every-7',32,7,4,True),('real/untrained-failures',64,500,5,False),('real/original-horizon-tail',64,500,6,True)]:
 n=net if trained else ActorCritic(seed=0);b=collect_rollout(n,CartPole(seed=seed,max_steps=max_steps),horizon,np.random.default_rng(seed+100));x={k:b[k].tolist() for k in ['rewards','values','next_values','terminated','truncated']};x.update(gamma=.99,lam=.95,tail_value=b['tail_value']);r=gae_audit(x);a,ret=compute_gae_transitions(**{k:x[k] for k in ['rewards','values','next_values','terminated','truncated','gamma','lam']});np.testing.assert_allclose(a,[v['advantage'] for v in r['rows']],atol=1e-12)
 fixtures.append(dict(name=name,kind='gae',input=x,report=r,capture=b,source='Actual unchanged CartPole physics and '+('shipped trained checkpoint' if trained else 'original seed-0 initialization')+'; all transitions retained; max_steps='+str(max_steps)))
def gae(name,r,v,nv,term,trunc,g=.9,l=.8,tail=0.):
 x=dict(rewards=r,values=v,next_values=nv,terminated=term,truncated=trunc,gamma=g,lam=l,tail_value=tail);fixtures.append(dict(name=name,kind='gae',input=x,report=gae_audit(x),source='Constructed boundary control, not a training trajectory'))
gae('control/reset-credit-leak',[1,1,100],[2,3,50],[3,7,4],[False]*3,[False,True,False],tail=4)
gae('control/true-termination',[1,1,100],[2,3,50],[3,7,4],[False,True,False],[False]*3,tail=4)
gae('control/tail-truncation',[1],[2],[7],[False],[True],tail=-9)
gae('control/discount-zero',[1,2],[3,4],[4,5],[False,False],[False,False],g=0,tail=5)
gae('control/lambda-zero',[1,2],[3,4],[4,5],[False,False],[False,False],l=0,tail=5)
gae('control/every-step-reset',[1,2,3],[3,4,5],[8,9,10],[False]*3,[True]*3,g=1,l=1,tail=-2)
for sign in [1,-1,0]:
 for ratio in [.5,.8,1,1.2,1.5]:
  z=[[0.,0.]];lp=-math.log(2);x=dict(logits=z,values=[.3],actions=[0],old_logp=[lp-math.log(ratio)],advantages=[sign*2.],returns=[1.],clip_eps=.2,vf_coef=.5,ent_coef=.01)
  r=ppo_audit(x);loss,dz,dv=ppo_loss_and_grads(z,x['values'],x['actions'],x['old_logp'],x['advantages'],x['returns'],clip_eps=.2,vf_coef=.5,ent_coef=.01)
  np.testing.assert_allclose(loss,r['loss'],rtol=1e-13,atol=1e-13);np.testing.assert_allclose(dz,[q['dlogits'] for q in r['rows']],atol=1e-13);np.testing.assert_allclose(dv,[q['dvalue'] for q in r['rows']],atol=1e-13)
  fixtures.append(dict(name=f'control/advantage-{sign}/ratio-{ratio}',kind='ppo',input=x,report=r,source='Constructed sign/branch control; old action log-probability is valid'))
# Complete actual original network loss batch, no cherry-picked row subset.
n=ActorCritic(hidden=5,seed=3);rng=np.random.default_rng(9);obs=rng.normal(size=(4,4));logits,values,_=n.forward(obs);actions=rng.integers(0,2,size=4);x=dict(logits=logits.tolist(),values=values.tolist(),actions=actions.tolist(),old_logp=log_softmax(logits)[np.arange(4),actions].tolist(),advantages=rng.normal(size=4).tolist(),returns=rng.normal(size=4).tolist(),clip_eps=.2,vf_coef=.5,ent_coef=.01)
fixtures.append(dict(name='real/seeded-tiny-network-loss',kind='ppo',input=x,report=ppo_audit(x),source='Complete four-row original ActorCritic(hidden=5,seed=3) batch; not original training minibatch'))
write('fixtures.json',fixtures)
# Every step of all twenty historical final-evaluation episodes, and all ten original gate episodes.
episodes=[]
for seed,count in [(12345,20),(987,10)]:
 env=CartPole(seed=seed)
 for episode in range(count):
  obs=env.reset();rows=[]
  for t in range(500):
   z,v,_=net.forward(obs[None,:]);a=int(np.argmax(z[0]));nobs,reward,term,trunc=env.step(a);rows.append(dict(t=t,observation=obs.tolist(),logits=z[0].tolist(),value=float(v[0]),action=a,next_observation=nobs.tolist(),reward=reward,terminated=term,truncated=trunc));obs=nobs
   if term or trunc:break
  episodes.append(dict(seed=seed,episode=episode,rows=rows,return_=len(rows)))
write('model.json',dict(parameters=net.state_dict(),checkpoint_sha256=hashlib.sha256((ROOT/'assets/policy.npz').read_bytes()).hexdigest(),episodes=[dict(seed=e['seed'],episode=e['episode'],initial=e['rows'][0]['observation'],expected_return=e['return_'],expected_terminated=e['rows'][-1]['terminated'],expected_truncated=e['rows'][-1]['truncated']) for e in episodes],historical_results=json.loads((ROOT/'assets/results.json').read_text())))
# Browser data excludes redundant reports (native witnesses remain in wheel).
js=dict(replay_reference=json.loads((out/'replay-reference.json').read_text()),fixtures=[{k:v for k,v in f.items() if k!='report'} for f in fixtures],model=json.loads((out/'model.json').read_text()))
(ROOT/'docs/data.js').write_text('globalThis.PPO_DATA='+json.dumps(js,default=conv,separators=(',',':'),allow_nan=False)+';\n')
print(json.dumps(dict(fixtures=len(fixtures),transitions=sum(len(f['input'].get('rewards',[])) for f in fixtures),episodes=len(episodes),physics_steps=sum(len(e['rows']) for e in episodes),returns={str(seed):[e['return_'] for e in episodes if e['seed']==seed] for seed in [12345,987]})))
