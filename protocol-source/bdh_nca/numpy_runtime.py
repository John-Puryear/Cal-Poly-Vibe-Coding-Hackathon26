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
    x,ef,es,delay=state[:4];c=config;m=c["message"]
    induction=state[4] if c.get("explicit_induction_state",False) else None
    offsets=[(-1,0),(1,0),(0,-1),(0,1),(0,-c["long_radius"]),(0,c["long_radius"])]
    current=np.clip(control[:,:1],-1,1);block=np.clip(control[:,1:2],0,1)
    long_block=np.clip(control[:,2:3],0,1) if control.shape[1]>2 else np.zeros_like(block)
    purine_input=np.clip(control[:,3:4],0,1) if control.shape[1]>3 else np.zeros_like(block)
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
    long_open=np.stack([shift(1-long_block,*o) for o in offsets[4:]],1)[:,:,0]
    ws=np.concatenate((ws[:,:4],ws[:,4:]*(1-long_block)*long_open),1)
    lw=ws[:,:4]/np.maximum(1,ws[:,:4].sum(1,keepdims=True))
    rw=ws[:,4:]/np.maximum(1,ws[:,4:].sum(1,keepdims=True))
    if not c["long_range"]:rw=rw*0
    local=(lw[:,:,None]*np.stack([shift(value,*o) for o in offsets[:4]],1)).sum(1)
    distant=c["long_gain"]*(rw[:,:,None]*delay[:,:,:m]).sum(1)
    sensed=nb[:,:4].copy();eb=1-(1-block)*source_open[:,:4]
    sensed[:,:,4:8]-=eb[:,:,None]*(sensed[:,:,4:8]-x[:,None,4:8])
    n,s,w,e=[sensed[:,i] for i in range(4)];mean=(n+s+w+e)*.25
    cross=np.tanh(affine(x[:,4:8],weights,"fast_factor"))*np.tanh(affine(x[:,12:16],weights,"slow_factor"))
    if not c["interaction"]:cross=cross*0
    feat=np.concatenate((x,mean,(e-w)*.5,(s-n)*.5,mean-x,local,distant,cross),1)
    if c.get("control_sensing",False):
        feat=np.concatenate((feat,current,block,long_block),1)
    if induction is not None:
        feat=np.concatenate((feat,induction[:,:2]),1)
    delta=np.tanh(affine(np.maximum(affine(feat,weights,"update.0"),0),weights,"update.2"))*weights["rates"]
    if c.get("temporal_face_gate",False):
        ready=np.clip((abs(x[:,4:5])-.12)/.32,0,1)
        gate_face=1-np.clip(x[:,23:24],0,1)*(1-ready)
        delta=np.concatenate((delta[:,:1],delta[:,1:3]*gate_face,delta[:,3:12],
                              delta[:,12:13]*gate_face,delta[:,13:14],
                              delta[:,14:15]*gate_face,delta[:,15:]),1)
    ldv=(lw*(nb[:,:4,4]-x[:,4:5])).sum(1,keepdims=True)
    rdv=(rw*(delay[:,:,m]-x[:,4:5])).sum(1,keepdims=True)
    fb=.015*x[:,21:22]*(1-np.clip(x[:,15:16],0,1)) if c["consolidation"] else x[:,4:5]*0
    vextra=.12*ldv+.025*rdv+.15*current+fb
    if c.get("octanol_voltage_repatterning",False):
        deprived=block*induction[:,2:3]*np.clip(-x[:,13:14],0,1)
        recalled=np.clip(induction[:,:1]-.10,0,1)
        vextra=vextra+(.06*deprived+.35*recalled)*(.55-x[:,4:5])
    extra=np.concatenate((np.zeros_like(x[:,:4]),vextra,np.zeros_like(x[:,5:20])),1)
    dyn=np.clip(x[:,:20]+fire*(delta+extra-.002*x[:,:20]),-1,1)
    if c.get("induction_memory",False):
        purine=purine_input if induction is not None else np.clip(x[:,8:9]-.5,0,1)*2
        gate=np.maximum(np.maximum(block,long_block),np.maximum(abs(current),purine))
        writer_input=np.concatenate((x[:,0:1],x[:,4:9],x[:,12:14],current,block,long_block),1)
        writer_drive=np.tanh(affine(np.maximum(affine(writer_input,weights,"memory_writer.0"),0),weights,"memory_writer.2"))
        if c.get("injury_provenance",False):
            head_stimulus=np.maximum(block,long_block)*induction[:,2:3]
            eye_stimulus=np.maximum(abs(current),purine)
            memory_delta=.02*writer_drive*np.concatenate((head_stimulus,eye_stimulus),1)*fire
            if c.get("axial_default_head",False):
                released=.025*block*induction[:,2:3]*np.clip(-x[:,13:14],0,1)*fire
                memory_delta=np.concatenate((memory_delta[:,:1]+released,memory_delta[:,1:2]),1)
        else:
            memory_delta=.02*writer_drive*gate*fire
        if induction is not None:
            memory_state=induction[:,:2]
            bistable=.01*memory_state*(1-memory_state)*(memory_state-.25)*fire
            bounded=np.clip(memory_state+memory_delta+bistable,0,1)
            induction_next=np.concatenate((bounded,induction[:,2:3]*.995),1) if c.get("injury_provenance",False) else bounded
        else:
            dyn=np.concatenate((dyn[:,:16],np.clip(dyn[:,16:18]+memory_delta,-1,1),dyn[:,18:20]),1)
    if c.get("morphogenetic_readout",False):
        source_memory=induction[:,:2] if induction is not None else x[:,16:18]
        latent_gate=np.clip(np.maximum(np.maximum(source_memory[:,0:1]-.15,0),np.maximum(source_memory[:,1:2]-.02,0)),0,1)
        edge_gate=np.clip(4*(ef[:,:4]+es[:,:4]).mean(1,keepdims=True),0,1)
        morph_features=np.concatenate((x,mean,ef+es,induction[:,:2]),1) if induction is not None else np.concatenate((x,mean,ef+es),1)
        morph_rate=.16 if induction is not None else .08
        if c.get("distinct_cascades",False):
            edges=(ef+es)[:,:4]
            hi=[0,1,3,4,7,12,13,18,20,22];ei=[0,2,4,5,8,12,14,19,21,23]
            hf=np.concatenate((x[:,hi],mean[:,hi],edges,induction[:,:1],x[:,7:8],x[:,13:14]),1)
            eftr=np.concatenate((x[:,ei],mean[:,ei],edges,induction[:,1:2],x[:,5:6],x[:,8:9]),1)
            head_threshold=c.get("axial_head_threshold",.15)
            hg=np.clip(induction[:,:1]-head_threshold,0,1)*edge_gate*fire
            eye_signal=np.clip(induction[:,1:2]-.02,0,1)
            if c.get("compositional_eye_recruitment",False):
                organizer=np.clip(induction[:,:1]-head_threshold,0,1)*np.clip(dyn[:,1:2],0,1)
                eye_signal=np.maximum(eye_signal,.8*organizer)
            eg=eye_signal*edge_gate*fire
            hd=morph_rate*np.tanh(affine(np.maximum(affine(hf,weights,"head_cascade.0"),0),weights,"head_cascade.2"))*hg
            ed=morph_rate*np.tanh(affine(np.maximum(affine(eftr,weights,"eye_cascade.0"),0),weights,"eye_cascade.2"))*eg
            dyn=np.concatenate((dyn[:,:1],np.clip(dyn[:,1:2]+hd[:,:1],-1,1),
                np.clip(dyn[:,2:3]+ed[:,:1],-1,1),np.clip(dyn[:,3:4]+hd[:,1:2],-1,1),
                np.clip(dyn[:,4:5]+hd[:,2:3],-1,1),dyn[:,5:12],
                np.clip(dyn[:,12:13]+ed[:,1:2],-1,1),np.clip(dyn[:,13:14]+hd[:,3:4],-1,1),
                np.clip(dyn[:,14:15]+ed[:,2:3],-1,1),dyn[:,15:20]),1)
        else:
            morph_delta=morph_rate*np.tanh(affine(np.maximum(affine(morph_features,weights,"morph_readout.0"),0),weights,"morph_readout.2"))*latent_gate*edge_gate*fire
            dyn=np.concatenate((dyn[:,:1],np.clip(dyn[:,1:4]+morph_delta[:,:3],-1,1),
                                np.clip(dyn[:,4:5]+morph_delta[:,3:4],-1,1),dyn[:,5:12],
                                np.clip(dyn[:,12:15]+morph_delta[:,4:7],-1,1),dyn[:,15:20]),1)
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
    if c.get("loop_reinforcement",False):
        source_memory=induction[:,:2] if induction is not None else x[:,16:18]
        memory=np.clip(np.maximum(source_memory,0).sum(1,keepdims=True),0,1)
        neighbor_memory=np.stack([shift(memory,*o) for o in offsets],1)[:,:,0]
        edge=ef+es
        reciprocal=np.stack((shift(edge[:,1:2],-1,0),shift(edge[:,0:1],1,0),
                             shift(edge[:,3:4],0,-1),shift(edge[:,2:3],0,1),
                             shift(edge[:,5:6],0,-c["long_radius"]),
                             shift(edge[:,4:5],0,c["long_radius"])),1)[:,:,0]
        loop_drive=memory*neighbor_memory*agreement*structural*(1-x[:,15:16])
        eligible=np.maximum(eligible+np.tanh(np.maximum(weights["loop_gain"],0))*loop_drive*(.2+.8*reciprocal),0)
    eo=(1-block)*source_open
    eo=np.concatenate((eo[:,:4],eo[:,4:]*(1-long_block)*long_open),1)
    eligible=eligible*eo
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
    if induction is not None:
        induction_next=induction_next*frontier(x[:,:1])*frontier(cells[:,:1])
        return cells,nf,ns,nd,induction_next
    return cells,nf,ns,nd

def validate(folder):
    meta,w=load_weights(folder);f=np.load(Path(folder)/"conformance.npz",allow_pickle=False)
    state=tuple(f[f"input_{i}"] for i in range(len(meta["state_order"])));error=0.0
    for tick in range(meta["fixture_ticks"]):
        state=step(state,f[f"control_{tick}"],f[f"fire_{tick}"],w,meta["config"])
        for i,t in enumerate(state):error=max(error,float(np.max(abs(t-f[f"output_{tick}_{i}"]))))
    if error>1e-5:raise AssertionError(f"NumPy/PyTorch mismatch: {error}")
    return error

if __name__=="__main__":print(json.dumps({"numpy_recurrent_max_error":validate(sys.argv[1])}))
