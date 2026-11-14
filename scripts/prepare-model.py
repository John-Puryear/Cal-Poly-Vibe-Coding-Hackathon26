"""Copy portable model data into the website; never modify the source package."""
from pathlib import Path
import sys, json, base64, shutil, hashlib
import numpy as np

source = Path(sys.argv[1])
project = Path(__file__).resolve().parents[1]
out = project / 'dist' / 'model'
out.mkdir(parents=True, exist_ok=True)
portable=source/'runs'/'demo'/'portable'
for name in ('manifest.json','weights.f32'):
    shutil.copyfile(portable/name, out/name)
dataset=np.load(source/'synthetic_data'/'test.npz',allow_pickle=False)
print('Dataset fields:', dataset.files)
initial=dataset['precursor']
items={}
for kind,ch in [('face',23),('planaria',22)]:
    indexes=[i for i in range(len(initial)) if initial[i,ch].max()>.1]
    for index,i in enumerate(indexes):
        a=initial[i].astype('<f4')
        items[f'{kind}-{index}']={'kind':kind,'width':a.shape[2],'height':a.shape[1],
            'cells':base64.b64encode(a.tobytes()).decode('ascii'),'source_index':i,'split':'test'}
(out/'initial-states.json').write_text(json.dumps(items,separators=(',',':')))
fixture=np.load(portable/'conformance.npz',allow_pickle=False)
data={k:{'shape':list(fixture[k].shape),'data':fixture[k].ravel().tolist()} for k in fixture.files}
(project/'tests'/'fixture.json').write_text(json.dumps(data,separators=(',',':')))
metadata={'source':'bdh_nca_demo/runs/demo/portable','weights_sha256':hashlib.sha256((out/'weights.f32').read_bytes()).hexdigest(),
          'initial_states':'Supplied synthetic test precursors; no final targets are loaded by the runtime.',
          'model_status':'Synthetic normal-field training. Alternative anatomical phenotypes remain unestablished.'}
(out/'provenance.json').write_text(json.dumps(metadata,indent=2))
print('Prepared',len(items),'test precursors and conformance fixtures.')
