"""Deterministic original arithmetic controls, not sampled training evidence."""
import json
from pathlib import Path
from ppolab.clipface import audit
from ppolab.clipverify import verify
CASES=[
 ('asymmetric-centered','Clipping is not a hard ratio box',['1/10','9/10'],[9,-1],'1/5'),
 ('symmetric-centered','A whole optimal plateau',['1/2','1/2'],[-1,1],'1/5'),
 ('zero-advantage','No incentive is not a proximity bound',['1/2','1/3','1/6'],[0,0,0],'1/5'),
 ('forced-zero','No full-support exact optimum',['1/3']*3,[1,2,3],'4/5'),
 ('equal-positive','Identical positive slopes',['1/4']*4,[1]*4,'1/5'),
 ('zero-epsilon','Zero clipping width is still not a hard constraint',['1/10','9/10'],[9,-1],'0'),
 ('mixed-three','Three unequal signed contributions',['1/6','1/3','1/2'],[3,-2,1],'1/4'),
 ('six-actions','Six-action face and tied slopes',['1/21','2/21','3/21','4/21','5/21','6/21'],[1,-2,0,3,-1,3],'1/5'),
 ('rare-action','A rare action and exact ratios',['1/1000','999/1000'],[999,-1],'1/5'),
 ('negative','Negative supplied advantages are not silently centered',['1/2','1/2'],[-1,-2],'1/5'),
]
def generate():
 out=[]
 for id,title,p,a,e in CASES:
  x={'version':1,'old':p,'advantages':a,'epsilon':e};r=audit(p,a,e);verify(r)
  out.append({'id':id,'title':title,'input':x,'report':r})
 return out
if __name__=='__main__':
 data=generate();text=json.dumps(data,sort_keys=True,separators=(',',':'))+'\n'
 Path('src/ppolab/data/clip-fixtures.json').write_text(text)
 Path('docs/clip-data.js').write_text('globalThis.PPO_CLIP_FIXTURES='+text.rstrip()+';\n')
 print(len(data),'independently replayed exact controls')
