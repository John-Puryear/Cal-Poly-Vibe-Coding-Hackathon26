"""Transparent screening metrics. Connected components are not anatomical proof."""
import numpy as np

def components(mask,min_area=2):
    mask=np.asarray(mask,dtype=bool); seen=np.zeros_like(mask);out=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        todo=[(int(y),int(x))];seen[y,x]=True;pts=[]
        while todo:
            a,b=todo.pop();pts.append((a,b))
            for c,d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0<=c<mask.shape[0] and 0<=d<mask.shape[1] and mask[c,d] and not seen[c,d]:
                    seen[c,d]=True;todo.append((c,d))
        if len(pts)>=min_area:
            out.append({"area":len(pts),"y":float(np.mean([p[0] for p in pts])),"x":float(np.mean([p[1] for p in pts]))})
    return out


def measure(cells):
    a=cells.detach().cpu().numpy()[0]
    live=a[0]>.35
    heads=components((a[1]>.45)&live,min_area=3)
    eyes=components((a[2]>.35)&live,min_area=1)
    return {"head_domains":heads,"eye_domains":eyes,"head_count":len(heads),
            "eye_count":len(eyes),"tissue_area":int(live.sum()),
            "max_abs_state":float(np.abs(a).max()),"voltage_std":float(a[4][live].std()) if live.any() else 0.0}
