"""Normal-only pool training. Interventions live exclusively in evaluate.py."""
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
    pool=model.initial(seeds.clone());ages=[0]*args.pool
    history=[];start=time.perf_counter()
    for iteration in range(1,args.iterations+1):
        ids=random.sample(range(args.pool),args.batch)
        batch=TissueState(*(t[ids].clone() for t in pool.tensors()))
        for j,i in enumerate(ids):
            if ages[i]>=args.pool_chunks or j==0:
                reset=model.initial(seeds[i:i+1])
                for t,r in zip(batch.tensors(),reset.tensors()):t[j:j+1]=r
                ages[i]=0
        if args.damage and iteration>args.iterations//3:
            # Optional explicit repair training; recorded in checkpoint provenance.
            hole=torch.ones_like(batch.cells[:1,:1]);h=args.size
            yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(h,device=device),indexing="ij")
            cy,cx=random.randrange(h),random.randrange(h)
            hole[0,0]=(((yy-cy)**2+(xx-cx)**2)>(h*.12)**2)
            batch=model.damage(batch,hole)
        ticks=random.randint(args.steps,args.steps+args.steps//2)
        for _ in range(ticks):batch=model(batch)
        loss=loss_fields(batch.cells,targets[ids])
        loss=loss+1e-5*batch.cells[:,8:12].abs().mean()+1e-5*batch.edge_fast.mean()
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
                 "elapsed_seconds":time.perf_counter()-start}
            history.append(row);print(json.dumps(row),flush=True)
            sheet([("Normal target",val.target),("Precursor",val.seed),(f"Model: step {iteration}",vs.cells)],out/"training_preview.png")
            checkpoint={"metadata":model.metadata(),"model":model.state_dict(),
                        "training":{"iterations":iteration,"seed":args.seed,"size":args.size,
                                    "damage_training":args.damage,"pool":args.pool,
                                    "initial_checkpoint_sha256":initial_hash,"args":vars(args)},
                        "history":history}
            torch.save(checkpoint,out/"checkpoint.pt")
            (out/"history.json").write_text(json.dumps(history,indent=2))
    return model
