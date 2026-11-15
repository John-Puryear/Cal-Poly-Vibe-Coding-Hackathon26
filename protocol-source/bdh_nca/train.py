"""Normal-anatomy pool training with optional isolated physiology exercises."""
from dataclasses import asdict
from pathlib import Path
import json
import random
import time
import hashlib
import numpy as np
import torch
from .data import make_example
from .model import BDHNCA, Config, TissueState
from .render import sheet

SUPERVISED = list(range(8))+list(range(12,16))

def resolve_device(name):
    if name=="auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return name

def loss_fields(x,target):
    # Rare eye/head/tail pixels get extra weight without an abnormal target.
    weight=x.new_tensor([1,3,8,3,3,1,1,.2,1,.5,.5,.2])[None,:,None,None]
    spatial=.2+target[:,:1]+8*target[:,2:3]
    return ((x[:,SUPERVISED]-target[:,SUPERVISED]).square()*weight*spatial).mean()

def train(args):
    torch.manual_seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device=resolve_device(args.device)
    config=Config(hidden=args.hidden,plasticity=not args.no_plasticity,
                  consolidation=not args.no_consolidation,
                  interaction=not args.no_interaction,long_range=not args.no_long_range)
    model=BDHNCA(config).to(device)
    initial_hash=None
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location=device,weights_only=True)
        if initial["metadata"]["config"]!=asdict(config):
            raise ValueError("Warm-start configuration must match the checkpoint")
        model.load_state_dict(initial["model"])
        initial_hash=hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    # Each pool entry remains paired to its own normal target.
    if args.data:
        data=np.load(args.data,allow_pickle=False)
        if data["full_target"].shape[0]<args.pool or data["full_target"].shape[2:]!=(args.size,args.size):
            raise ValueError("Dataset must contain >= pool examples at the requested size")
        targets=torch.from_numpy(data["full_target"][:args.pool]).to(device)
        seeds=torch.from_numpy(data["precursor"][:args.pool]).to(device)
    else:
        examples=[make_example(args.size,args.seed*10000+i) for i in range(args.pool)]
        targets=torch.cat([e.target for e in examples]).to(device)
        seeds=torch.cat([e.seed for e in examples]).to(device)
    if args.principles:
        from .model import shift
        for i in range(args.pool):
            if float(targets[i,22].max())<.5:continue
            # Isolated normal anterior fragments: never two heads or extra eyes
            # on an intact animal. Translated fragments discourage absolute location.
            if i%8==0:
                mask=(torch.arange(args.size,device=device)[None,:]<args.size*.49)[None,None]
                t=targets[i:i+1]*mask
                t=shift(t,random.randint(-4,4),-random.randint(0,10))
                targets[i:i+1]=t
            if i%8 in (0,2):
                seeds[i]=targets[i]*0
                seeds[i,0]=targets[i,0]*.85
                seeds[i,4:8]=targets[i,4:8]*.85
                seeds[i,14]=targets[i,0]*.25
                seeds[i,22:24]=targets[i,22:24]
    if args.principles:
        np.savez_compressed(out/"curriculum.npz",precursor=seeds.cpu().numpy(),full_target=targets.cpu().numpy())
    if args.xenopus:
        from .xenopus import bank,auxiliary_loss
        xi,xt=bank(args.size);np.savez_compressed(out/"eye_primordia.npz",precursor=xi.numpy(),next_state=xt.numpy(),transition_ticks=8)
        xi,xt=xi.to(device),xt.to(device)
    pool=model.initial(seeds.clone());ages=[0]*args.pool
    history=[];start=time.perf_counter();trunk_exposures=0
    for iteration in range(1,args.iterations+1):
        ids=random.sample(range(args.pool),args.batch)
        # Classic NCA: reseed the worst entry, damage the best mature entries.
        ids.sort(key=lambda i: float(loss_fields(pool.cells[i:i+1],targets[i:i+1]).detach()),reverse=True)
        batch=TissueState(*(t[ids].clone() for t in pool.tensors()))
        for j,i in enumerate(ids):
            if ages[i]>=args.pool_chunks or j==0:
                reset=model.initial(seeds[i:i+1])
                for t,r in zip(batch.tensors(),reset.tensors()):t[j:j+1]=r
                ages[i]=0
        if args.damage:
            h=args.size
            yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(h,device=device),indexing="ij")
            keep=torch.ones_like(batch.cells[:,:1])
            damage_count=max(1,args.batch//3)
            for j in range(max(1,args.batch-damage_count),args.batch):
                if ages[ids[j]] < 1:continue
                if (args.trunk_curriculum and j==args.batch-1 and
                    targets[ids[j],22].amax()>.5 and random.random()<.5):
                    keep[j,0]=(xx>=10)&(xx<=23)
                    trunk_exposures+=1
                    continue
                sites=torch.nonzero(targets[ids[j],0]>.5)
                cy,cx=sites[random.randrange(len(sites))]
                radius=random.uniform(.07,.16)*h
                candidate=((yy-cy)**2+(xx-cx)**2)>radius**2
                if int(((batch.cells[j,0]>.1)&candidate).sum())>=3:
                    keep[j,0]=candidate
            batch=model.damage(batch,keep)
        ticks=random.randint(args.steps,args.steps+args.steps//2)
        for _ in range(ticks):batch=model(batch)
        loss=loss_fields(batch.cells,targets[ids])
        # Penalize uncontrolled growth outside the normal target scaffold.
        loss=loss+.5*(batch.cells[:,:1]*(1-targets[ids,:1])).square().mean()
        loss=loss+1e-5*batch.cells[:,8:12].abs().mean()+1e-5*batch.edge_fast.mean()
        if args.xenopus:
            loss=loss+.2*auxiliary_loss(model,xi,xt)
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError("Non-finite training update")
        opt.step()
        batch=batch.detach()
        for dst,src in zip(pool.tensors(),batch.tensors()):dst[ids]=src
        for i in ids:ages[i]+=1
        if iteration==1 or iteration%args.log_every==0 or iteration==args.iterations:
            with torch.no_grad():
                val=make_example(args.size,17,kind="planaria",split="validation")
                vs=model.initial(val.seed.to(device))
                for _ in range(args.eval_steps):vs=model(vs)
                vl=float(loss_fields(vs.cells,val.target.to(device)))
            row={"iteration":iteration,"train_loss":float(loss.detach()),"validation_loss":vl,
                 "normal_trunk_exposures":trunk_exposures,
                 "elapsed_seconds":time.perf_counter()-start}
            history.append(row);print(json.dumps(row),flush=True)
            sheet([("Normal target",val.target),("Precursor",val.seed),(f"Model: step {iteration}",vs.cells)],out/"training_preview.png")
            checkpoint={"metadata":model.metadata(),"model":model.state_dict(),
                        "training":{"iterations":iteration,"seed":args.seed,"size":args.size,
                                    "damage_training":args.damage,"pool":args.pool,
                                    "damage_fraction":max(1,args.batch//3)/args.batch if args.damage else 0,
                                    "normal_trunk_curriculum":args.trunk_curriculum,
                                    "normal_trunk_exposures":trunk_exposures,
                                    "principles_revision":"balanced-v2" if args.principles else None,
                                    "xenopus_revision":"window-calcium-purine-fast8-v2" if args.xenopus else None,
                                    "initial_checkpoint_sha256":initial_hash,"args":vars(args)},
                        "history":history}
            torch.save(checkpoint,out/"checkpoint.pt")
            (out/"history.json").write_text(json.dumps(history,indent=2))
    return model
