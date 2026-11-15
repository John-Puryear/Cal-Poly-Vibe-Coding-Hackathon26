"""Normal-only synthetic developmental fields; no abnormal-phenotype branch.

These are invented dimensionless patterns, not measured biological data.
Geometry exists only in this offline generator, never in model.forward.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch

@dataclass
class Example:
    seed: torch.Tensor
    target: torch.Tensor
    kind: str
    landmarks: dict


def gaussian(u,v,a,b,sx,sy):
    return np.exp(-.5*((u-a)/sx)**2-.5*((v-b)/sy)**2)


def make_example(size=32, seed=0, kind=None, split="train"):
    if size < 16:
        raise ValueError("Use a grid of at least 16 cells per side")
    if split not in ("train","validation","test"):
        raise ValueError("Unknown split")
    # Disjoint RNG namespaces; evaluation never consumes the training generator.
    rng=np.random.default_rng(seed+{"train":0,"validation":10**7,"test":2*10**7}[split])
    kind=kind or ("planaria" if rng.random()<.5 else "face")
    if kind not in ("planaria","face"):
        raise ValueError("Only normal planaria and normal face are training domains")
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    angle=rng.uniform(-.18,.18)
    cx,cy=rng.uniform(-.07,.07,2)
    u=(x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
    v=-(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
    scale_u=rng.uniform(.91,1.03);scale_v=rng.uniform(.91,1.03)
    u/=scale_u;v/=scale_v
    target=np.zeros((24,size,size),np.float32)
    if kind=="planaria":
        body=(u/.78)**2+(v/(.28+.045*np.tanh(-3*u)))**2 < 1
        head=gaussian(u,v,-.49,0,.16,.20)
        eyes=np.maximum(gaussian(u,v,-.52,-.11,.045,.045),gaussian(u,v,-.52,.11,.045,.045))
        tail=gaussian(u,v,.60,0,.12,.13)
        voltage=.65*head-.65*eyes-.35*np.clip(u,0,1)
        starter=.30*gaussian(u,v,-.49,0,.30,.27)-.20*u
        polarity=-u
        landmarks={"head_uv":[-.49,0],"eye_uv":[[-.52,-.11],[-.52,.11]],"normal_head_count":1,"normal_eye_count":2}
    else:
        body=(u/.64)**2+(v/.77)**2 < 1
        eyes=np.maximum(gaussian(u,v,-.23,-.20,.065,.065),gaussian(u,v,.23,-.20,.065,.065))
        # Normal craniofacial territory and ventral domain; no rendered RGB image.
        head=gaussian(u,v,0,-.24,.46,.34)
        mouth=gaussian(u,v,0,.30,.26,.06)
        tail=np.zeros_like(head)
        voltage=.65*head-.95*eyes-.55*mouth
        starter=.25*gaussian(u,v,0,-.30,.48,.30)-.10*v
        polarity=-v
        landmarks={"eye_uv":[[-.23,-.20],[.23,-.20]],"normal_eye_count":2}
    target[0]=body
    target[1]=head*body;target[2]=eyes*body;target[3]=tail*body
    target[4]=voltage*body;target[5]=head*body;target[6]=(.3+.4*(1-head))*body
    target[12]=np.tanh(2*voltage)*body;target[13]=polarity*body
    target[14]=body*.75;target[15]=body*.15
    target[22+(kind=="face")]=body
    start=np.zeros_like(target)
    start[0]=body*.85
    start[4]=starter*body;start[5]=body*.1
    start[12]=starter*body*.5;start[13]=polarity*body
    start[14]=body*.25;start[15]=body*.05;start[22:]=target[22:]
    start[8:12]=rng.normal(0,.025,(4,size,size))*body
    landmarks.update({"cx":float(cx),"cy":float(cy),"angle":float(angle),
                      "scale_u":float(scale_u),"scale_v":float(scale_v)})
    return Example(torch.from_numpy(start)[None],torch.from_numpy(target)[None],kind,landmarks)


def dataset_audit(count=128,size=32,split="train",alternating=False):
    """Audit semantic construction and thresholded components, not just filenames."""
    from .metrics import components
    kinds={"planaria":0,"face":0}
    for i in range(count):
        e=make_example(size,i,kind=("planaria" if i%2==0 else "face") if alternating else None,split=split)
        eyes=components(e.target[0,2].numpy()>.35,min_area=1)
        heads=components(e.target[0,1].numpy()>.4,min_area=1)
        if len(eyes)!=2 or len(heads)!=1:
            raise AssertionError(f"Invalid normal example {i}: {len(heads)} territories, {len(eyes)} eyes")
        kinds[e.kind]+=1
    return {"split":split,"samples":count,"kinds":kinds,"two_head_targets":0,
            "ectopic_eye_targets":0,"normal_eye_components":2,
            "note":"Procedural construction audit; training rollouts may make errors."}
