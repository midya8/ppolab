#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$PYTHON" -m venv --system-site-packages "$TMP/venv"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$TMP/venv/bin/python" --quiet --no-deps "$ROOT" --target "$TMP/site"
else
  "$TMP/venv/bin/python" -m pip install --quiet --no-deps "$ROOT" --target "$TMP/site"
fi
cd "$TMP"
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" - <<'PY'
import importlib,json,math,pathlib
from importlib import resources
import numpy as np
mods=['ppolab','ppolab.ppo','ppolab.nets','ppolab.cartpole','ppolab.optim','ppolab.train','ppolab.audit','ppolab.boundaries']
for m in mods:
 x=importlib.import_module(m)
 assert '/site/' in x.__file__,x.__file__
from ppolab.audit import gae_audit,ppo_audit
from ppolab.nets import ActorCritic
from ppolab.cartpole import CartPole
from ppolab.boundaries import collect_rollout,compute_gae_transitions
from ppolab.train import evaluate,update
from ppolab.optim import Adam
p=resources.files('ppolab').joinpath('data');fixtures=json.loads(p.joinpath('fixtures.json').read_text());model=json.loads(p.joinpath('model.json').read_text())
for f in fixtures:
 r=(gae_audit if f['kind']=='gae' else ppo_audit)(f['input'])
 assert r==f['report']
net=ActorCritic();net.load_state_dict(model['parameters']);m,sd,returns=evaluate(net,seed=12345,episodes=20)
assert returns==[e['expected_return'] for e in model['episodes'] if e['seed']==12345]
assert abs(m-model['historical_results']['final_eval_mean_20ep'])<1e-9
net=ActorCritic(hidden=5,seed=0);b=collect_rollout(net,CartPole(seed=0,max_steps=3),32,np.random.default_rng(1));a,r=compute_gae_transitions(b['rewards'],b['values'],b['next_values'],b['terminated'],b['truncated']);assert sum(b['truncated'])>=10
assert math.isfinite(update(net,Adam(net.p),b['observations'],b['actions'],b['old_logp'],a,r,epochs=1,minibatch=8,rng=np.random.default_rng(2)))
for kind in ['gae','ppo']:
 f=next(f for f in fixtures if f['kind']==kind);pathlib.Path(kind+'.json').write_text(json.dumps(f['input']))
print('Installed artifact: eight module origins, all 26 full audits, full original 20-episode evaluation and corrected short-timeout training update verified')
PY
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m ppolab.audit gae --input gae.json > gae-report.json
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m ppolab.audit ppo --input ppo.json > ppo-report.json
PYTHONPATH="$TMP/site" "$TMP/venv/bin/python" -m ppolab.train --iters 1 --horizon 32 --out smoke > smoke.log
printf '%s\n' 'Both JSON CLI commands and the real training CLI ran outside checkout.'
