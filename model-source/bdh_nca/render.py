"""Layer previews; displayed anatomy is a readout of predicted fields."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def panel(cells, scale=5):
    a=cells.detach().cpu().numpy()[0] if hasattr(cells,"detach") else np.asarray(cells)
    live=a[0].clip(0,1)
    body=np.stack((.50+.25*a[1],.65+.18*a[1]-.25*a[3],.40+.20*a[3]),-1).clip(0,1)
    eye=a[2].clip(0,1)[...,None]
    body=body*(1-eye*.95)
    body=body*live[...,None]+(1-live[...,None])*.97
    v=a[4].clip(-1,1)
    voltage=np.stack((np.maximum(v,0)+.15,.25+.50*(1-abs(v)),np.maximum(-v,0)+.15),-1).clip(0,1)
    voltage=voltage*live[...,None]+(1-live[...,None])*.97
    gene=np.stack(((a[12]+1)*.5,(a[13]+1)*.5,a[14].clip(0,1)),-1).clip(0,1)
    gene=gene*live[...,None]+(1-live[...,None])*.97
    return [Image.fromarray((im*255).astype("uint8")).resize((a.shape[2]*scale,a.shape[1]*scale),Image.Resampling.NEAREST) for im in (body,voltage,gene)]


def sheet(rows,path,scale=5,title="BDH-NCA | synthetic layered tissue"):
    """rows: (label, tensor)."""
    first=panel(rows[0][1],scale);w,h=first[0].size
    canvas=Image.new("RGB",(3*w+200,len(rows)*(h+30)+65),(248,248,245));draw=ImageDraw.Draw(canvas)
    draw.text((15,12),title,fill=(25,35,35))
    for i,t in enumerate(("Body / head / eyes / tail","Voltage: red + / blue -","Gene / axis / structure")):
        draw.text((190+i*w,40),t,fill=(40,55,55))
    for j,(label,state) in enumerate(rows):
        yy=65+j*(h+30);draw.text((12,yy+10),label,fill=(30,45,45))
        for i,im in enumerate(panel(state,scale)):canvas.paste(im,(190+i*w,yy))
    Path(path).parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def animation(frames,path,scale=5):
    images=[]
    for x in frames:
        ps=panel(x,scale);w,h=ps[0].size
        im=Image.new("RGB",(w*3,h));[im.paste(p,(i*w,0)) for i,p in enumerate(ps)];images.append(im)
    images[0].save(path,save_all=True,append_images=images[1:],duration=100,loop=0)


def layer_details(example,path,scale=7):
    a=example.target.numpy()[0];h,w=a.shape[-2:];tile=w*scale
    fields=[("Body",a[0],False),("Head",a[1],False),("Eyes",a[2],False),("Tail",a[3],False),
            ("Voltage",a[4],True),("Regulatory",a[12],True),("Polarity",a[13],True),("Structure",a[14],False)]
    canvas=Image.new("RGB",(tile*4,2*(h*scale+38)+45),(248,248,245));d=ImageDraw.Draw(canvas)
    d.text((12,12),"Normal 2D planarian | aligned synthetic data layers",fill=(25,35,35))
    for i,(label,f,signed) in enumerate(fields):
        if signed:
            rgb=np.stack((np.maximum(f,0),.20+.45*(1-abs(f)),np.maximum(-f,0)),-1).clip(0,1)
        else:rgb=np.repeat(f.clip(0,1)[...,None],3,-1)
        im=Image.fromarray((rgb*255).astype("uint8")).resize((tile,h*scale),Image.Resampling.NEAREST)
        xx=(i%4)*tile;yy=45+(i//4)*(h*scale+38)
        d.text((xx+8,yy),label,fill=(30,45,45));canvas.paste(im,(xx,yy+22))
    canvas.save(path)
