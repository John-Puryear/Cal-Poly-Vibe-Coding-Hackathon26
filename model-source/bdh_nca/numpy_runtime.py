"""Independent inference reference using only NumPy, suitable for a JS port.

Run directly: python bdh_nca/numpy_runtime.py runs/demo/portable
No PyTorch operation or training code is used in this file.
"""
from pathlib import Path
import json
import sys
import numpy as np

def shift(x,dy,dx):
    h,w=x.shape[-2:];pads=[(0,0)]*(x.ndim-2)+[(max(-dy,0),max(dy,0)),(max(-dx,0),max(dx,0))]
    p=np.pad(x,pads)
    return p[...,max(dy,0):max(dy,0)+h,max(dx,0):max(dx,0)+w]

def load_weights(folder):
    p=Path(folder);meta=json.loads((p/"manifest.json").read_text());raw=(p/"weights.f32").read_bytes()
    weights={k:np.frombuffer(raw,dtype='<f4',count=v["byte_length"]//4,offset=v["byte_offset"]).reshape(v["shape"]) for k,v in meta["weights"].items()}
    return meta,weights

def affine(x,w,name):
    kernel=w[name+".weight"][:,:,0,0]
    return np.einsum('oi,bihw->bohw',kernel,x,optimize=True)+w[name+".bias"][None,:,None,None]

def sigmoid(x):return 1/(1+np.exp(-x))

def frontier(alpha):
    return np.maximum.reduce([shift(alpha,dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)])>.1

def step(state,control,fire,weights,config):
    x,ef,es,delay=state;c=config;m=c["message"]
    offsets=[(-1,0),(1,0),(0,-1),(0,1),(0,-c["long_radius"]),(0,c["long_radius"])]
    current=np.clip(control[:,:1],-1,1);block=np.clip(control[:,1:2],0,1)
    control=np.concatenate((current,block),1)
    alive=(x[:,:1]>.1).astype(np.float32)
    q=np.tanh(affine(x,weights,"query"));k=np.tanh(affine(x,weights,"key"));value=np.tanh(affine(x,weights,"value"))
    nb=np.stack([shift(x,*o) for o in offsets],1)
    edge_alive=alive[:,None]*(nb[:,:,:1]>.1)
    edge_alive=np.concatenate((edge_alive[:,:4],edge_alive[:,4:]*delay[:,:,-1:]),1)
    keys=np.stack([shift(k,*o) for o in offsets],1)
    score=sigmoid((q[:,None]*keys).sum(2)/(m**.5))
    coupling=.20+.45*ef+.35*es if c["plasticity"] else np.full_like(ef,.20)
    ws=score*coupling*edge_alive[:,:,0]
    source_open=np.stack([shift(1-block,*o) for o in offsets],1)[:,:,0]
    ws=ws*(1-block)*source_open
    lw=ws[:,:4]/np.maximum(1,ws[:,:4].sum(1,keepdims=True))
    rw=ws[:,4:]/np.maximum(1,ws[:,4:].sum(1,keepdims=True))
    if not c["long_range"]:rw=rw*0
    local=(lw[:,:,None]*np.stack([shift(value,*o) for o in offsets[:4]],1)).sum(1)
    distant=c["long_gain"]*(rw[:,:,None]*delay[:,:,:m]).sum(1)
    n,s,w,e=[nb[:,i] for i in range(4)];mean=(n+s+w+e)*.25
    cross=np.tanh(affine(x[:,4:8],weights,"fast_factor"))*np.tanh(affine(x[:,12:16],weights,"slow_factor"))
    if not c["interaction"]:cross=cross*0
    feat=np.concatenate((x,mean,(e-w)*.5,(s-n)*.5,mean-x,local,distant,cross),1)
    delta=np.tanh(affine(np.maximum(affine(feat,weights,"update.0"),0),weights,"update.2"))*weights["rates"]
    ldv=(lw*(nb[:,:4,4]-x[:,4:5])).sum(1,keepdims=True)
    rdv=(rw*(delay[:,:,m]-x[:,4:5])).sum(1,keepdims=True)
    fb=.015*x[:,21:22]*(1-np.clip(x[:,15:16],0,1)) if c["consolidation"] else x[:,4:5]*0
    vextra=.12*ldv+.025*rdv+.15*current+fb
    extra=np.concatenate((np.zeros_like(x[:,:4]),vextra,np.zeros_like(x[:,5:20])),1)
    dyn=np.clip(x[:,:20]+fire*(delta+extra-.002*x[:,:20]),-1,1)
    dyn=np.concatenate((np.clip(dyn[:,:4],0,1),dyn[:,4:15],np.clip(dyn[:,15:16],0,1),dyn[:,16:20]),1)
    concordance=np.clip(x[:,4:5]*x[:,12:13],0,1)
    trace=x[:,20:21]+.04*fire*(x[:,4:5]-x[:,20:21])
    pref=x[:,21:22]+.003*fire*(concordance*trace-x[:,21:22])
    if not c["consolidation"]:trace=trace*0;pref=pref*0
    cells=np.concatenate((dyn,np.clip(trace,-1,1),np.clip(pref,-1,1),x[:,22:24]),1)
    cells=cells*frontier(x[:,:1])*frontier(cells[:,:1])
    agreement=np.clip(1-abs(nb[:,:,4]-x[:,4:5])*.5,0,1)
    structural=np.clip(1-abs(nb[:,:,12]-x[:,12:13])*.5,0,1)
    eligible=agreement*structural*abs(x[:,4:5])*(1-x[:,15:16])
    transfer=.003*ef if c["consolidation"] else ef*0
    nf=np.clip(ef+.06*eligible*np.maximum(1-ef-es,0)/(1+ef.sum(1,keepdims=True))-.02*ef-transfer,0,1)
    ns=np.clip(es+transfer-.0005*es,0,1)
    scale=np.maximum(1,nf+ns);nf=nf/scale;ns=ns/scale
    nalive=(cells[:,:1]>.1).astype(np.float32)
    valid=np.concatenate([nalive*shift(nalive,*o) for o in offsets],1)
    enabled=np.array([1,1,1,1,int(c["long_range"]),int(c["long_range"])],np.float32)[None,:,None,None]
    nf=nf*valid*enabled;ns=ns*valid*enabled
    if not c["plasticity"]:nf=nf*0;ns=ns*0
    if not c["consolidation"]:ns=ns*0
    packet=np.concatenate((value,x[:,4:5],alive),1)
    nd=np.stack([shift(packet,*o) for o in offsets[4:]],1)*nalive[:,None]*enabled[:,4:,None]
    return cells,nf,ns,nd

def validate(folder):
    meta,w=load_weights(folder);f=np.load(Path(folder)/"conformance.npz",allow_pickle=False)
    state=tuple(f[f"input_{i}"] for i in range(4));error=0.0
    for tick in range(meta["fixture_ticks"]):
        state=step(state,f[f"control_{tick}"],f[f"fire_{tick}"],w,meta["config"])
        for i,t in enumerate(state):error=max(error,float(np.max(abs(t-f[f"output_{tick}_{i}"]))))
    if error>1e-5:raise AssertionError(f"NumPy/PyTorch mismatch: {error}")
    return error

if __name__=="__main__":print(json.dumps({"numpy_recurrent_max_error":validate(sys.argv[1])}))
