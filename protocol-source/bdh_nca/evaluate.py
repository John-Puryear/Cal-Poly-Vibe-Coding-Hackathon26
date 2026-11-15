"""Unseen electrical interventions. Never constructs an abnormal anatomy target."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import make_example
from .metrics import measure
from .model import BDHNCA, Config, TissueState
from .render import sheet, animation
from .train import resolve_device, loss_fields

def load(path,device="cpu"):
    ck=torch.load(path,map_location=device,weights_only=True)
    model=BDHNCA(Config(**ck["metadata"]["config"])).to(device)
    model.load_state_dict(ck["model"]);model.eval()
    return model,ck

def controls(cells,scenario,strength=.8):
    """Spatial voltage current and conductance intervention only; no fate writes."""
    _,_,h,w=cells.shape
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=cells.device),torch.linspace(-1,1,w,device=cells.device),indexing="ij")
    c=cells.new_zeros(1,2,h,w)
    if scenario=="tail_pulse":
        region=((x-.52)/.17)**2+(y/.18)**2<1
        c[:,0]=region*strength;c[:,1]=region*.7
    elif scenario=="ectopic_pulse":
        region=((x-.12)/.10)**2+(y/.10)**2<1
        c[:,0]=-region.to(cells.dtype)*strength
    return c

def erase(model,state):
    x=state.cells.clone();h,w=x.shape[-2:]
    y,z=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing="ij")
    keep=(((y-h*.5)**2+(z-w*.42)**2)>(h*.12)**2)[None,None]
    return model.damage(state,keep)

def screen(control,treated,normal_target):
    """Require both normal competence and retained existing structures."""
    c=measure(control);t=measure(treated)
    target=normal_target[0].cpu().numpy()
    normal_head=(target[1]>.35);normal_eyes=(target[2]>.25)
    a=treated[0].detach().cpu().numpy()
    cc=control[0].detach().cpu().numpy()
    competent=(c["head_count"]==1 and c["eye_count"]==2 and
               float(cc[1][normal_head].mean())>.40 and float(cc[2][normal_eyes].mean())>.30)
    retained=float(a[1][normal_head].mean())>.40 and float(a[2][normal_eyes].mean())>.30
    h,w=target.shape[-2:]
    tail=[p for p in t["head_domains"] if p["x"]>w*.62]
    ectopic=[p for p in t["eye_domains"] if p["x"]>w*.45]
    return {"normal_competence_gate":bool(competent),"retained_normal_structures":bool(retained),
            "two_head_candidate":bool(competent and retained and t["head_count"]==2 and tail),
            "ectopic_eye_candidate":bool(competent and retained and t["eye_count"]>2 and ectopic),
            "interpretation":"Post-washout field screening only; inspect anatomy, stability, and seed replication."}

@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    device=resolve_device(args.device);model,ck=load(args.checkpoint,device)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    results={"checkpoint_training":ck["training"],"config":model.metadata(),"cases":[],
             "thresholds":{"head":.45,"eye":.35,"head_min_area":3,"eye_min_area":1},
             "status":"Exploratory held-out tests; no guaranteed emergent phenotype"}
    for seed in range(args.seeds):
        ex=make_example(args.size,seed,kind="planaria",split="test")
        final={};rows=[("Normal target",ex.target)]
        for scenario in ("control","tail_pulse","ectopic_pulse","damage"):
            # Same initial state and random update schedule in paired experiments.
            torch.manual_seed(100000+seed)
            state=model.initial(ex.seed.to(device));frames=[state.cells.cpu()]
            stages={}
            total=args.develop+args.pulse+args.washout
            for tick in range(total):
                if tick==args.develop and scenario=="damage":state=erase(model,state)
                c=controls(state.cells,scenario,args.strength) if args.develop<=tick<args.develop+args.pulse else None
                state=model(state,c)
                if tick in (args.develop-1,args.develop+args.pulse-1,total-1):
                    stages[str(tick+1)]=measure(state.cells)
                if tick%4==0 or tick==total-1:frames.append(state.cells.cpu())
            final[scenario]=state.cells.cpu();rows.append((f"{scenario}: washout",state.cells))
            np.savez_compressed(out/f"seed{seed}_{scenario}.npz",cells=state.cells.cpu().numpy())
            if seed==0:animation(frames,out/f"{scenario}.gif",scale=5)
            results["cases"].append({"seed":seed,"scenario":scenario,"stages":stages,
                "normal_field_loss":float(loss_fields(state.cells,ex.target.to(device))),
                "edge_fast_mean":float(state.edge_fast.mean()),"edge_slow_mean":float(state.edge_slow.mean())})
        for row in results["cases"][-4:]:
            row["screen"]=screen(final["control"],final[row["scenario"]],ex.target)
        sheet(rows,out/f"seed{seed}_comparison.png",title="Held-out tests | targets are normal-only; outcomes are model predictions")
    # A normal face is a separate developmental domain, with the same shared rule.
    ex=make_example(args.size,0,"face","test");state=model.initial(ex.seed.to(device))
    torch.manual_seed(12345)
    for _ in range(args.develop+args.pulse+args.washout):state=model(state)
    sheet([("Normal target",ex.target),("Precursor",ex.seed),("Predicted face",state.cells)],out/"face.png")
    results["face"]={"metrics":measure(state.cells),"normal_field_loss":float(loss_fields(state.cells,ex.target.to(device)))}
    results["protocol"]={"develop":args.develop,"pulse":args.pulse,"washout":args.washout,"strength":args.strength,"seeds":args.seeds}
    (out/"metrics.json").write_text(json.dumps(results,indent=2))
    print(json.dumps({"out":str(out),"cases":len(results["cases"]),"status":results["status"]}),flush=True)
