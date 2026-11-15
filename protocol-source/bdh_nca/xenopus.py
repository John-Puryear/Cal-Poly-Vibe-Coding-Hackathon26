"""Invented eye-primordium dynamics inspired by primary Xenopus studies.

Not a fitted Xenopus simulator. Teacher exists only in auxiliary training.
Never called by BDHNCA.forward or by held-out intact-worm inference.
"""
import numpy as np
import torch
from .model import shift

def teacher_tick(x,drive,availability,purine):
    y=x.clone();body=x[:,:1];v=x[:,4:5];ca=x[:,5:6];p=x[:,8:9];tf=x[:,16:17];rx=x[:,17:18]
    def lap(z):return sum(shift(z,*o) for o in ((-1,0),(1,0),(0,-1),(0,1)))*.25-z
    window=torch.exp(-((v+.35)/.12).square())
    y[:,4:5]=(v+.02*(drive-v)+.06*tf*(-.35-v)+.04*lap(v)).clamp(-1,1)
    y[:,5:6]=(ca+.15*(availability*window-ca)+.03*lap(ca)).clamp(0,1)
    y[:,8:9]=(p+.08*(purine+.15*tf-p)+.06*lap(p)).clamp(0,1)
    target=torch.sigmoid(12*(ca+.9*p+1.1*tf-.65))
    y[:,16:17]=(tf+.0075*(target-tf)).clamp(0,1)
    y[:,17:18]=(rx+.0075*(tf-rx)).clamp(0,1)
    y[:,2:3]=(x[:,2:3]+.06*(tf*rx-x[:,2:3])).clamp(0,1)
    y[:,12:13]=x[:,12:13]+.0075*(v-x[:,12:13])
    y[:,14:15]=x[:,14:15]+.005*(.5+.3*rx-x[:,14:15])
    y[:,18:19]=x[:,18:19]+.0075*(ca-x[:,18:19])
    trace=x[:,20:21]+.02*(v-x[:,20:21]);y[:,20:21]=trace
    y[:,21:22]=x[:,21:22]+.0015*((v*x[:,12:13]).clamp(0,1)*trace-x[:,21:22])
    return y*body

@torch.no_grad()
def bank(size=32,seed=413):
    rng=np.random.default_rng(seed);yy,xx=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    configs=[(-.35,1.,.05),(-.05,1.,.05),(-.75,1.,.05),(-.35,0.,.05),(0.,1.,.9),(-.35,1.,.05)]*2
    initial=[]
    for i,(v,available,purine) in enumerate(configs):
        x=torch.zeros(1,24,size,size);cy,cx=rng.integers(8,size-8,2);radius=int(rng.integers(3,6))
        body=(((yy-int(cy))**2+(xx-int(cx))**2)<radius**2).float()[None,None]
        x[:,0:1]=body;x[:,4:5]=body*v;x[:,8:9]=body*(purine if purine>.5 else .05);x[:,14:15]=body*.5
        x[:,19:20]=body*(0 if available else -1);x[:,22+i%2:23+i%2]=body
        initial.append(x)
    x=torch.cat(initial);inputs=[];outputs=[]
    for tick in range(256):
        if tick%8==0:inputs.append(x.clone())
        drive=x.new_tensor([a[0] if tick<128 else 0 for a in configs])[:,None,None,None]
        available=x.new_tensor([a[1] for a in configs])[:,None,None,None]
        purine=x.new_tensor([a[2] if tick<128 else .05 for a in configs])[:,None,None,None]
        x=teacher_tick(x,drive,available,purine)
        if tick%8==7:outputs.append(x.clone())
    # Only one isolated tissue/primordium per example, never an intact ectopic eye.
    return torch.cat(inputs),torch.cat(outputs)

def auxiliary_loss(model,inputs,targets,batch=2):
    ids=torch.randint(len(inputs),(batch,),device=inputs.device)
    s=model.initial(inputs[ids])
    for _ in range(8):s=model(s)
    channels=[0,1,2,3,4,5,8,12,14,16,17,18,19,20,21]
    weights=s.cells.new_tensor([1,1,4,1,2,2,1,1,.5,3,3,1,.5,.2,.2])[None,:,None,None]
    mask=.1+targets[ids,:1]
    return ((s.cells[:,channels]-targets[ids][:,channels]).square()*weights*mask).mean()
