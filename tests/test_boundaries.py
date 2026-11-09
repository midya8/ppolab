import copy,hashlib,json,subprocess,sys
from pathlib import Path
from decimal import Decimal,localcontext
import numpy as np
import pytest
from ppolab.audit import gae_audit,ppo_audit,read_json,validate_gae
from ppolab.boundaries import collect_rollout,compute_gae_transitions
from ppolab.cartpole import CartPole
from ppolab.nets import ActorCritic
from ppolab.ppo import ppo_loss_and_grads,compute_gae
from ppolab.train import rollout,update
ROOT=Path(__file__).resolve().parents[1]
FIX=json.loads((ROOT/'src/ppolab/data/fixtures.json').read_text())


def test_original_files_and_results_preserved():
    hashes=json.loads((ROOT/'src/ppolab/data/original-hashes.json').read_text())
    for path,digest in hashes.items():
        if path=='src/ppolab/train.py':continue
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert len([p for p in hashes if p.startswith('tests/')])==5


def test_all_gae_witnesses_against_numpy_and_explicit_sums():
    for f in FIX:
        if f['kind']!='gae':continue
        x=f['input'];r=gae_audit(x);assert r==f['report'];a,ret=compute_gae_transitions(**{k:x[k] for k in ('rewards','values','next_values','terminated','truncated','gamma','lam')})
        np.testing.assert_allclose(a,[q['advantage'] for q in r['rows']],atol=1e-12)
        np.testing.assert_allclose(ret,[q['target'] for q in r['rows']],atol=1e-12)
        for i,row in enumerate(r['contributions']):
            assert sum(row)==pytest.approx(a[i],abs=1e-11)
            for j in range(i+1,len(row)):
                if any(x['terminated'][i:j]) or any(x['truncated'][i:j]):assert row[j]==0


def test_independent_80_digit_decimal_gae_reference():
    with localcontext() as ctx:
        ctx.prec=80
        d=lambda x:Decimal(str(x))
        for f in FIX:
            if f['kind']!='gae':continue
            x=f['input'];a=[]
            for t in range(len(x['rewards'])):
                total=Decimal(0);w=Decimal(1)
                for j in range(t,len(x['rewards'])):
                    delta=d(x['rewards'][j])+d(x['gamma'])*d(x['next_values'][j])*(not x['terminated'][j])-d(x['values'][j]);total+=w*delta
                    if x['terminated'][j] or x['truncated'][j]:break
                    w*=d(x['gamma'])*d(x['lam'])
                a.append(float(total))
            np.testing.assert_allclose(a,[r['advantage'] for r in gae_audit(x)['rows']],rtol=1e-12,atol=1e-12)


def test_reset_reward_cannot_leak_backwards():
    f=next(f for f in FIX if f['name']=='control/reset-credit-leak');x=copy.deepcopy(f['input']);before=gae_audit(x);x['rewards'][2]=-100;after=gae_audit(x)
    assert [r['advantage'] for r in before['rows'][:2]]==[r['advantage'] for r in after['rows'][:2]]
    assert before['rows'][1]['old_reset_advantage']!=after['rows'][1]['old_reset_advantage']
    assert before['max_old_error']>1 and before['max_terminal_error']>1


def test_single_step_truncation_uses_actual_not_reset_value():
    x=next(f['input'] for f in FIX if f['name']=='control/tail-truncation');r=gae_audit(x)['rows'][0]
    assert r['advantage']==pytest.approx(1+.9*7-2)
    assert r['old_reset_advantage']==pytest.approx(1+.9*(-9)-2)
    assert r['wrong_terminal_advantage']==-1


def test_actual_collector_and_legacy_adapter_agree():
    net=ActorCritic(seed=0)
    for steps in [1,3,7,500]:
        b=collect_rollout(net,CartPole(seed=3,max_steps=steps),64,np.random.default_rng(100))
        tup=rollout(net,CartPole(seed=3,max_steps=steps),64,np.random.default_rng(100),gamma=.9)
        a,ret=compute_gae_transitions(b['rewards'],b['values'],b['next_values'],b['terminated'],b['truncated'],gamma=.9,lam=.8)
        a2,r2=compute_gae(*[tup[k] for k in [3,5,4,6]],gamma=.9,lam=.8)
        np.testing.assert_allclose(a,a2,atol=1e-12);np.testing.assert_allclose(ret,r2,atol=1e-12)
        np.testing.assert_array_equal(tup[4],b['terminated']|b['truncated'])
        assert np.all(b['rewards']==1)
        for i in range(64):
            _,v,_=net.forward(b['next_observations'][i]);assert b['next_values'][i]==v[0]


def test_real_capture_matches_collector():
    for f,steps,seed,trained in zip(FIX[:4],[3,7,500,500],[3,4,5,6],[True,True,False,True]):
        net=ActorCritic.load(ROOT/'assets/policy.npz') if trained else ActorCritic(seed=0)
        b=collect_rollout(net,CartPole(seed=seed,max_steps=steps),len(f['input']['rewards']),np.random.default_rng(seed+100))
        for key,val in f['capture'].items():np.testing.assert_allclose(b[key],val,atol=1e-12)


def test_all_ppo_witnesses_against_unchanged_numpy_loss():
    for f in FIX:
        if f['kind']!='ppo':continue
        x=f['input'];r=ppo_audit(x);assert r==f['report']
        loss,dz,dv=ppo_loss_and_grads(x['logits'],x['values'],x['actions'],x['old_logp'],x['advantages'],x['returns'],clip_eps=x['clip_eps'],vf_coef=x['vf_coef'],ent_coef=x['ent_coef'])
        assert loss==pytest.approx(r['loss'],abs=1e-13);np.testing.assert_allclose(dz,[q['dlogits'] for q in r['rows']],atol=1e-13);np.testing.assert_allclose(dv,[q['dvalue'] for q in r['rows']],atol=1e-13)
        assert r['max_smooth_error']<1e-8
        for q in r['rows']:assert sum(q['dlogits'])==pytest.approx(0,abs=1e-12)


def test_negative_advantage_outside_can_stay_active():
    f=next(f for f in FIX if f['name']=='control/advantage--1/ratio-1.5');q=f['report']['rows'][0];assert q['ratio_outside'] and q['via_unclipped'];assert q['dlogits'][0]!=0
    f=next(f for f in FIX if f['name']=='control/advantage-1/ratio-1.5');assert not f['report']['rows'][0]['via_unclipped']


def test_clip_kinks_are_not_false_gradient_failures():
    checks=[c for f in FIX if f['kind']=='ppo' for c in f['report']['checks']]
    kinks=[c for c in checks if not c['smooth']];assert len(kinks)>=4
    for c in kinks:assert c['normalized_error'] is None;assert min(c['left'],c['right'])-1e-4<=c['analytic']<=max(c['left'],c['right'])+1e-4


@pytest.mark.parametrize('bad',['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}', '['*10+'0'+']'*10,'{} extra','"'+('x'*100001)+'"'])
def test_strict_json_rejects_bad_input(bad):
    with pytest.raises(ValueError):read_json(bad)


@pytest.mark.parametrize('mutation',[lambda x:x.update(extra=1),lambda x:x.update(gamma=True),lambda x:x['terminated'].__setitem__(0,1),lambda x:x['next_values'].__setitem__(0,999),lambda x:x.update(lam=-1),lambda x:x.update(values=[]),lambda x:x['truncated'].__setitem__(0,True)])
def test_bad_gae_schema(mutation):
    x=copy.deepcopy(next(f['input'] for f in FIX if f['name']=='control/true-termination'))
    # Last mutation is made conflicting with an explicit termination.
    mutation(x)
    if x['truncated'][0]:x['terminated'][0]=True
    with pytest.raises(ValueError):gae_audit(x)


@pytest.mark.parametrize('key,val',[('clip_eps',True),('vf_coef',-1),('old_logp',[1]),('actions',[True]),('logits',[[0]]),('returns',[]),('ent_coef',2)])
def test_bad_ppo_schema(key,val):
    x=copy.deepcopy(FIX[10]['input']);x[key]=val
    with pytest.raises(ValueError):ppo_audit(x)


def test_cli_error_has_no_json():
    p=subprocess.run([sys.executable,'-m','ppolab.audit','gae','--input','-'],input='{"x":1,"x":2}',text=True,capture_output=True)
    assert p.returncode==2 and p.stdout=='' and 'duplicate' in p.stderr


def test_corrected_training_update_smoke():
    from ppolab.optim import Adam
    net=ActorCritic(hidden=5,seed=0);b=collect_rollout(net,CartPole(seed=0,max_steps=3),32,np.random.default_rng(1));a,r=compute_gae_transitions(b['rewards'],b['values'],b['next_values'],b['terminated'],b['truncated']);before=net.state_dict();loss=update(net,Adam(net.p),b['observations'],b['actions'],b['old_logp'],a,r,epochs=2,minibatch=8,rng=np.random.default_rng(2));assert np.isfinite(loss);assert any(np.any(before[k]!=net.p[k]) for k in before)
