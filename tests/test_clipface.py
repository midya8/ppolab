"""Independent finite references for the exact categorical objective extension."""
from fractions import Fraction as F
from itertools import product, permutations
from copy import deepcopy
import json
import math
import subprocess
import sys
import pytest
from ppolab.clipface import audit, problem, rational, term
from ppolab.clipverify import verify


def test_exhaustive_signed_advantages_against_independent_cell_replay():
    # 351 full optimizer reports; replay never imports/calls the optimizer.
    total = 0
    for n in [2, 3, 4]:
        for adv in product([-1, 0, 1], repeat=n):
            for eps in ['0', '1/5', '4/5']:
                assert verify(audit([str(F(1,n))]*n, list(adv), eps))
                total += 1
    assert total == 351


def test_asymmetric_centered_control_has_strict_hard_box_gap():
    r = audit(['1/10','9/10'], [9,-1], '1/5')
    assert r['old_mean_advantage'] == '0'
    f = r['simplex']; b = r['ratio_box']
    assert f['objective'] == '9/25' and b['objective'] == '1/5'
    assert r['hard_box_objective_gap'] == '4/25'
    assert f['ranges'] == [['7/25','1'],['0','18/25']]
    assert f['minimum_tv'] == '9/50' and f['maximum_tv'] == '9/10'
    assert f['full_support_kl_unbounded'] is True
    assert b['full_support_kl_unbounded'] is False
    for point in f['positive_optimal_approach']:
        assert min(map(F,point['policy']))>0 and point['objective']==f['objective']
    assert verify(r)


def test_plateau_extrema_and_nonindependent_ranges():
    r = audit(['1/2','1/2'],[-1,1],'1/5')['simplex']
    assert r['ranges']==[['0','2/5'],['3/5','1']]
    assert r['minimum_tv']=='1/10' and r['maximum_tv']=='1/2'
    assert r['vertices']==[['0','1'],['2/5','3/5']]
    assert sum(F(x[1]) for x in r['ranges']) != 1


def test_zero_advantages_simplex_and_support_loss():
    r = audit(['1/2','1/3','1/6'],[0,0,0],'1/5')
    assert r['simplex']['objective']=='0'
    assert r['simplex']['nearest']==r['old']
    assert r['simplex']['maximum_tv']=='5/6'
    assert len(r['simplex']['vertices'])==3
    assert verify(r)


def test_forced_zero_has_no_full_support_optimum():
    r = audit(['1/3']*3,[1,2,3], '4/5')['simplex']
    # No mean-centering requirement: this is a supplied surrogate, not an
    # inferred true advantage. Higher-slope actions take all available mass.
    assert r['forced_zero']==[0]
    assert not r['full_support_optimum_exists']
    assert not r['full_support_kl_unbounded']
    assert r['old_to_new_kl_infinite_on_closed_face']


def test_grid_policies_never_exceed_optimum_and_face_membership():
    p=['1/6','1/3','1/2'];a=[3,-2,1];e='1/4';r=audit(p,a,e)
    pp,aa,ee=problem(p,a,e)
    for i in range(25):
        for j in range(25-i):
            q=[F(i,24),F(j,24),1-F(i+j,24)]
            value=sum(term(x,y,ee,z) for x,y,z in zip(pp,aa,q))
            f=r['simplex'];inside=all(F(l)<=x<=F(u) for x,(l,u) in zip(q,f['ranges']))
            assert value<=F(f['objective'])
            assert (value==F(f['objective']))==inside
    assert verify(r)


def test_six_actions_and_extreme_small_probabilities():
    p=['1/21','2/21','3/21','4/21','5/21','6/21']
    for a in [[1,-2,0,3,-1,3],[0]*6,[7]*6,[-7]*6]:
        assert verify(audit(p,a,'999/1000'))
    assert verify(audit(['1/1000000000000000000000000000000','999999999999999999999999999999/1000000000000000000000000000000'],[1,-1],'1/5'))


def test_action_permutation_and_positive_scale():
    p=['1/6','1/3','1/2'];a=[3,-2,1];r=audit(p,a,'1/5')
    for order in permutations(range(3)):
        t=audit([p[i] for i in order],[a[i] for i in order],'1/5')
        for domain in ['simplex','ratio_box']:
            assert t[domain]['objective']==r[domain]['objective']
            assert t[domain]['ranges']==[r[domain]['ranges'][i] for i in order]
            assert t[domain]['maximum_tv']==r[domain]['maximum_tv']
        assert verify(t)
    scaled=audit(p,[7*x for x in a],'1/5')
    assert F(scaled['simplex']['objective'])==7*F(r['simplex']['objective'])
    assert scaled['simplex']['ranges']==r['simplex']['ranges']
    # Baseline shifts are deliberately NOT asserted invariant under clipping.


def test_every_report_leaf_tamper_is_rejected():
    report=audit(['1/10','9/10'],[9,-1],'1/5');paths=[]
    def walk(x,path):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,path+[k])
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,path+[i])
        else:paths.append(path)
    walk(report,[])
    for path in paths:
        r=deepcopy(report);node=r
        for key in path[:-1]:node=node[key]
        value=node[path[-1]]
        node[path[-1]]=not value if isinstance(value,bool) else '1234567' if isinstance(value,str) else 2 if value is not None else []
        with pytest.raises(ValueError):verify(r)
    assert len(paths)>100
    for extra in [dict(report,extra=0),{k:v for k,v in report.items() if k!='simplex'}]:
        with pytest.raises(ValueError):verify(extra)


def test_original_numpy_loss_matches_full_old_policy_expectation():
    import numpy as np
    from ppolab.ppo import ppo_loss_and_grads
    p=[F(1,10),F(9,10)];a=[9,-1];eps=F(1,5)
    for q in [[F(1,10),F(9,10)],[F(7,25),F(18,25)],[F(99,100),F(1,100)]]:
        acts=np.array([0]+[1]*9);logits=np.tile(np.log(list(map(float,q))),(10,1))
        loss,_,_=ppo_loss_and_grads(logits,np.zeros(10),acts,np.log([float(p[i]) for i in acts]),np.array([a[i] for i in acts]),np.zeros(10),clip_eps=float(eps),vf_coef=0,ent_coef=0)
        expected=sum(term(pi,ai,eps,qi) for pi,ai,qi in zip(p,a,q))
        assert math.isclose(-loss,float(expected),abs_tol=1e-14)


def test_strict_inputs_and_no_mutation():
    for x in [True,False,1.0,None,[],{},'1/0','x','9'*101,'1/'+str(2**130)]:
        with pytest.raises(ValueError):rational(x)
    for p,a,e in [([],[],0),([1,0],[1,2],0),(['1/2','1/3'],[1,2],0),(['1/2']*2,[1],0),(['1/2']*2,[1,2],1),(['1/2']*2,[1,2],'-1/10')]:
        with pytest.raises(ValueError):audit(p,a,e)
    p=['1/2','1/2'];a=[1,-1];copy=(p[:],a[:]);r=audit(p,a,'1/5');assert (p,a)==copy
    assert verify(r)


def test_cli_roundtrip_and_refusals(tmp_path):
    f=tmp_path/'input.json';out=tmp_path/'report.json'
    payload={'version':1,'old':['1/10','9/10'],'advantages':[9,-1],'epsilon':'1/5'}
    f.write_text(json.dumps(payload));cmd=[sys.executable,'-m','ppolab.clipface',str(f)]
    result=subprocess.run(cmd,text=True,capture_output=True);assert result.returncode==0
    out.write_text(result.stdout);verify(json.loads(result.stdout))
    checked=subprocess.run([sys.executable,'-m','ppolab.clipverify',str(out)],text=True,capture_output=True);assert checked.returncode==0
    for text in ['{"version":1,"version":1}',json.dumps(dict(payload,extra=0)),json.dumps(dict(payload,version=True)),'x'*100001]:
        f.write_text(text);r=subprocess.run(cmd,text=True,capture_output=True);assert r.returncode==2 and not r.stdout


def test_packaged_fixture_and_browser_loader_regeneration():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('clip_fixtures',Path(__file__).resolve().parents[1]/'scripts/generate_clip_fixtures.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text=json.dumps(module.generate(),sort_keys=True,separators=(',',':'))+'\n'
    root=Path(__file__).resolve().parents[1]
    assert (root/'src/ppolab/data/clip-fixtures.json').read_text()==text
    assert (root/'docs/clip-data.js').read_text()=='globalThis.PPO_CLIP_FIXTURES='+text.rstrip()+';\n'


def test_additional_json_refusals_and_independent_module_boundary(tmp_path):
    from pathlib import Path
    import inspect
    import ppolab.clipverify as checker
    source=inspect.getsource(checker)
    assert 'from .clipface' not in source and 'import ppolab.clipface' not in source
    f=tmp_path/'bad.json'
    for command,limit in [('clipface',100001),('clipverify',2000001)]:
        for text in ['['*1500+'0'+']'*1500,'{"x":NaN}','x'*limit]:
            f.write_text(text);r=subprocess.run([sys.executable,'-m','ppolab.'+command,str(f)],capture_output=True,text=True)
            assert r.returncode==2 and not r.stdout
