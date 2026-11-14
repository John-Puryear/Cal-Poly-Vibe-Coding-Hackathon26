"""Portable float32 weights and deterministic recurrent conformance vectors."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import make_example
from .evaluate import load
from .model import TissueState

@torch.no_grad()
def export(args):
    model,ck=load(args.checkpoint)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    arrays={k:v.cpu().numpy() for k,v in model.state_dict().items()}
    offset=0;layout={};chunks=[]
    for name,a in arrays.items():
        a=a.astype('<f4');chunks.append(a.tobytes())
        layout[name]={"shape":list(a.shape),"byte_offset":offset,"byte_length":a.nbytes};offset+=a.nbytes
    (out/"weights.f32").write_bytes(b''.join(chunks))
    metadata=model.metadata()
    metadata.update({"dtype":"float32","endianness":"little","tensor_layout":"NCHW",
                     "weights":layout,"state_order":["cells","edge_fast","edge_slow","delayed"],
                     "source_training":ck["training"],"fixture_ticks":8,
                     "ops":["pointwise affine","ReLU","tanh","sigmoid","zero-padded shift",
                            "3x3 max pool","elementwise arithmetic","clamp","sum","concatenate"],
                     "randomness":"Supply the recorded binary fire masks; no RNG inside the portable tick."})
    (out/"manifest.json").write_text(json.dumps(metadata,indent=2))
    # JSON is convenient for an initial JS implementation; binary is smaller.
    (out/"weights.json").write_text(json.dumps({k:v.tolist() for k,v in arrays.items()}))
    ex=make_example(24,41,"planaria","test");state=model.initial(ex.seed)
    fixture={f"input_{i}":t.numpy() for i,t in enumerate(state.tensors())}
    torch.manual_seed(9001)
    for tick in range(8):
        fire=(torch.rand_like(state.cells[:,:1])<model.config.fire_rate).float()
        control=torch.zeros(1,2,24,24)
        if tick in (2,3):control[:,0,10:14,14:17]=.5
        fixture[f"fire_{tick}"]=fire.numpy();fixture[f"control_{tick}"]=control.numpy()
        state=model(state,control,fire)
        for i,t in enumerate(state.tensors()):fixture[f"output_{tick}_{i}"]=t.numpy()
    np.savez_compressed(out/"conformance.npz",**fixture)
    # Reload the binary rather than trusting serialization by inspection.
    raw=(out/"weights.f32").read_bytes();restored={}
    for name,info in layout.items():
        a=np.frombuffer(raw,dtype='<f4',count=info["byte_length"]//4,offset=info["byte_offset"])
        restored[name]=torch.from_numpy(a.copy().reshape(info["shape"]))
    from .model import BDHNCA, Config
    other=BDHNCA(Config(**metadata["config"]));other.load_state_dict(restored)
    state=other.initial(ex.seed)
    error=0.0
    for tick in range(8):
        state=other(state,torch.from_numpy(fixture[f"control_{tick}"]),torch.from_numpy(fixture[f"fire_{tick}"]))
        for i,t in enumerate(state.tensors()):error=max(error,float((t-torch.from_numpy(fixture[f"output_{tick}_{i}"])).abs().max()))
    from .numpy_runtime import validate
    numpy_error=validate(out)
    report={"binary_roundtrip_max_error":error,"numpy_recurrent_max_error":numpy_error,"ticks":8,"bytes":offset,
            "browser_runtime_tested":False,"required_browser_tolerance":1e-5}
    (out/"export_validation.json").write_text(json.dumps(report,indent=2))
    if error>1e-6:raise AssertionError(report)
    print(json.dumps(report))
