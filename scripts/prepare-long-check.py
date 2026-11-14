from pathlib import Path
import sys,json,runpy,base64
import numpy as np
source=Path(sys.argv[1]);project=Path(__file__).resolve().parents[1]
ref=runpy.run_path(str(source/'bdh_nca'/'numpy_runtime.py'))
meta,weights=ref['load_weights'](source/'runs'/'demo'/'portable')
examples=json.loads((project/'dist'/'model'/'initial-states.json').read_text())
cases=[]
for kind in ('face-0','planaria-0'):
    e=examples[kind];h=e['height'];w=e['width'];n=w*h
    cells=np.frombuffer(base64.b64decode(e['cells']),dtype='<f4').copy().reshape(1,24,h,w)
    state=(cells,np.zeros((1,6,h,w),np.float32),np.zeros((1,6,h,w),np.float32),np.zeros((1,2,10,h,w),np.float32))
    rng=np.random.default_rng(12345);masks=[];snapshots={}
    for tick in range(144):
        fire=(rng.random((1,1,h,w))<.5).astype(np.float32);masks.append(fire.ravel().tolist())
        control=np.zeros((1,2,h,w),np.float32)
        if 96<=tick<120:
            yy,xx=np.mgrid[:h,:w];region=(xx-w//2)**2+(yy-h//2)**2<=9
            control[0,0]=region*.8
        state=ref['step'](state,control,fire,weights,meta['config'])
        if tick in (95,119,143):snapshots[str(tick+1)]=[a.ravel().tolist() for a in state]
    cases.append({'specimen':kind,'width':w,'height':h,'masks':masks,'snapshots':snapshots})
(project/'tests'/'long-fixture.json').write_text(json.dumps(cases,separators=(',',':')))
print('Prepared reference rollouts: 96 development / 24 pulse / 24 washout, two tissue types.')
