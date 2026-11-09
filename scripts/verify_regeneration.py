"""Regenerate with numeric tolerances across supported NumPy/Python versions."""
import json,subprocess,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'src/ppolab/data/fixtures.json',ROOT/'src/ppolab/data/model.json',ROOT/'docs/data.js']
def load(path):
 s=path.read_text()
 if path.suffix=='.js':s=s.removeprefix('globalThis.PPO_DATA=').rstrip().removesuffix(';')
 return json.loads(s)
def eq(a,b,key='root'):
 if isinstance(b,bool) or b is None or isinstance(b,str):assert a==b,key
 elif isinstance(b,(int,float)):
  tol=1e-8 if key.split('.')[-1] in ('left','right','central','normalized_error','max_smooth_error') else 1e-10
  assert isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and abs(a-b)<=tol*max(1,abs(a),abs(b)),(key,a,b)
 elif isinstance(b,list):
  assert isinstance(a,list) and len(a)==len(b),key
  for i,(x,y) in enumerate(zip(a,b)):eq(x,y,key+'['+str(i)+']')
 else:
  assert set(a)==set(b),key
  for k in b:eq(a[k],b[k],key+'.'+k)
before={p:p.read_bytes() for p in paths};old={p:load(p) for p in paths}
try:
 subprocess.run([sys.executable,str(ROOT/'scripts/regenerate.py')],check=True)
 for p in paths:eq(load(p),old[p],str(p.relative_to(ROOT)))
 print('All regenerated fixtures, full checkpoint/initial states and browser data match within declared tolerances')
finally:
 for p,value in before.items():p.write_bytes(value)
