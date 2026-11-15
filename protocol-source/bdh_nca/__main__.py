import argparse
import json
from pathlib import Path
import numpy as np
from .model import BDHNCA, Config
from .data import make_example, dataset_audit
from .render import sheet, layer_details

def main():
    p=argparse.ArgumentParser(description="Compact BDH-NCA layered biological demo")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("inspect");a.add_argument("--hidden",type=int,default=96)
    a=sub.add_parser("data");a.add_argument("--out",default="data");a.add_argument("--size",type=int,default=32);a.add_argument("--count",type=int,default=64)
    a=sub.add_parser("train")
    for flag,default in [("iterations",2000),("size",32),("batch",4),("pool",64),("steps",24),("pool-chunks",16),("hidden",96),("seed",7),("threads",4),("log-every",50),("eval-steps",96)]:
        a.add_argument("--"+flag,type=int,default=default)
    a.add_argument("--lr",type=float,default=.001);a.add_argument("--device",default="auto");a.add_argument("--out",default="runs/default")
    a.add_argument("--data",help="Optional normal-only train.npz; otherwise generate a reproducible normal pool")
    a.add_argument("--init-checkpoint",help="Warm-start model weights with a fresh optimizer and pool")
    a.add_argument("--damage",action="store_true")
    a.add_argument("--trunk-curriculum",action="store_true",help="Occasionally train normal full-body recovery from a mature planarian trunk fragment")
    a.add_argument("--xenopus",action="store_true",help="Auxiliary isolated eye-primordium physiology exercises")
    a.add_argument("--principles",action="store_true",help="Normal isolated head fragments and transient physiological prepatterns")
    for flag in ("plasticity","consolidation","interaction","long-range"):a.add_argument("--no-"+flag,action="store_true")
    a=sub.add_parser("evaluate");a.add_argument("--checkpoint",required=True);a.add_argument("--out",default="runs/evaluation")
    for flag,default in [("size",32),("seeds",3),("develop",96),("pulse",24),("washout",128),("threads",4)]:a.add_argument("--"+flag,type=int,default=default)
    a.add_argument("--strength",type=float,default=.8);a.add_argument("--device",default="auto")
    a=sub.add_parser("export");a.add_argument("--checkpoint",required=True);a.add_argument("--out",default="runs/export")
    args=p.parse_args()
    if args.command=="inspect":
        m=BDHNCA(Config(hidden=args.hidden));print(json.dumps(m.metadata(),indent=2))
    elif args.command=="data":
        if args.count<2:p.error("Use --count >= 2")
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest={"units":"dimensionless synthetic proxies","splits":{}}
        rows=[]
        for split,count in (("train",args.count),("validation",max(2,args.count//8)),("test",max(2,args.count//8))):
            examples=[make_example(args.size,i,kind="planaria" if i%2==0 else "face",split=split) for i in range(count)]
            targets=np.concatenate([e.target.numpy() for e in examples]);seeds=np.concatenate([e.seed.numpy() for e in examples])
            np.savez_compressed(out/f"{split}.npz",physical=targets[:,:4],bioelectric=targets[:,4:5],
                biochemical=targets[:,5:8],genomic_equivalent=targets[:,12:14],
                cytoskeletal_equivalent=targets[:,14:15],adaptation=targets[:,15:16],
                context=targets[:,22:24],precursor=seeds,full_target=targets,
                kind=np.array([e.kind for e in examples]))
            manifest["splits"][split]={"samples":count,"file":f"{split}.npz","seed_namespace":split}
            manifest["splits"][split]["audit"]=dataset_audit(count,args.size,split,alternating=True)
            if split=="train":
                rows=[("Planaria precursor",examples[0].seed),("Planaria normal target",examples[0].target),
                      ("Face precursor",examples[1].seed),("Face normal target",examples[1].target)]
        manifest["physical_channels"]=["body","head","eyes","tail"]
        manifest["genomic_equivalent_channels"]=["regulatory_state","axial_polarity"]
        manifest["abnormal_targets"]="None: exactly one head territory and two normal eyes per example"
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2));sheet(rows,out/"layer_preview.png")
        layer_details(make_example(args.size,0,"planaria"),out/"planaria_layers.png")
        print(json.dumps(manifest,indent=2))
    elif args.command=="train":
        if args.batch<1 or args.pool<args.batch or min(args.iterations,args.steps,args.threads,args.pool_chunks,args.log_every,args.eval_steps)<1:p.error("Positive iteration sizes required; pool >= batch >= 1")
        from .train import train
        train(args)
    elif args.command=="evaluate":
        if min(args.seeds,args.develop,args.pulse,args.washout,args.threads)<1 or not 0<=args.strength<=1:p.error("Positive rollout lengths and strength [0,1] required")
        from .evaluate import evaluate
        evaluate(args)
    else:
        from .export import export
        export(args)

if __name__=="__main__":main()
