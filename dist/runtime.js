// Independent float32 browser implementation of bdh_nca/numpy_runtime.py.
// NCHW state; synchronous read/write buffers; no target anatomy at inference.
const f = Math.fround;
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const sigmoid = x => 1 / (1 + Math.exp(-x));

export function decodeWeights(manifest, buffer) {
  const weights = {};
  for (const [name, item] of Object.entries(manifest.weights)) {
    weights[name] = new Float32Array(buffer, item.byte_offset, item.byte_length / 4);
  }
  return weights;
}

export function initialState(cells, width, height) {
  const n = width * height;
  return { cells: new Float32Array(cells), edgeFast: new Float32Array(6*n),
    edgeSlow: new Float32Array(6*n), delayed: new Float32Array(20*n) };
}

export function cloneState(s) {
  return { cells: s.cells.slice(), edgeFast: s.edgeFast.slice(), edgeSlow: s.edgeSlow.slice(), delayed: s.delayed.slice() };
}

export function makeRng(seed) {
  let a = seed >>> 0;
  return () => { a += 0x6D2B79F5; let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

export class TissueRuntime {
  constructor(manifest, weights, width, height) {
    this.config = { ...manifest.config };
    this.weights = weights; this.width = width; this.height = height;
    this.n = width * height;
    const r = this.config.long_radius;
    this.offsets = [[-1,0],[1,0],[0,-1],[0,1],[0,-r],[0,r]];
    this.neighbors = this.offsets.map(([dy,dx]) => {
      const map = new Int32Array(this.n); map.fill(-1);
      for (let y=0;y<height;y++) for(let x=0;x<width;x++) {
        const yy=y+dy, xx=x+dx;
        if(yy>=0&&yy<height&&xx>=0&&xx<width) map[y*width+x]=yy*width+xx;
      } return map;
    });
  }

  affine(input, inputChannels, outputChannels, name) {
    const n=this.n, w=this.weights[name+'.weight'], b=this.weights[name+'.bias'];
    const out=new Float32Array(outputChannels*n);
    for(let o=0;o<outputChannels;o++) {
      const wo=o*inputChannels, oo=o*n;
      for(let p=0;p<n;p++) {
        let value=b[o];
        for(let i=0;i<inputChannels;i++) value += w[wo+i]*input[i*n+p];
        out[oo+p]=value;
      }
    } return out;
  }

  frontier(cells,p) {
    const width=this.width, height=this.height, y=(p/width)|0, x=p%width;
    for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
      const yy=y+dy,xx=x+dx;
      if(yy>=0&&yy<height&&xx>=0&&xx<width&&cells[yy*width+xx]>.1) return true;
    } return false;
  }

  step(state, control, fire) {
    const {cells:x,edgeFast:ef,edgeSlow:es,delayed:delay}=state;
    const n=this.n,c=this.config,m=c.message,neighbors=this.neighbors;
    if(m!==8||c.rank!==4) throw new Error('This browser runtime expects the delivered 8-message, rank-4 model.');
    const current=new Float32Array(n), block=new Float32Array(n);
    for(let p=0;p<n;p++) {current[p]=clip(control?.[p]??0,-1,1);block[p]=clip(control?.[n+p]??0,0,1);}
    const q=this.affine(x,24,m,'query'), k=this.affine(x,24,m,'key'), value=this.affine(x,24,m,'value');
    for(let i=0;i<q.length;i++){q[i]=Math.tanh(q[i]);k[i]=Math.tanh(k[i]);value[i]=Math.tanh(value[i]);}
    const weights=new Float32Array(6*n), lw=new Float32Array(4*n), rw=new Float32Array(2*n);
    for(let p=0;p<n;p++) {
      let localSum=0, distantSum=0;
      for(let e=0;e<6;e++) {
        const j=neighbors[e][p], index=e*n+p;
        if(j<0||x[p]<=.1||x[j]<=.1||(e>=4&&delay[((e-4)*10+9)*n+p]===0))continue;
        let dot=0;
        for(let z=0;z<m;z++)dot=f(dot+f(q[z*n+p]*k[z*n+j]));
        const coupling=c.plasticity?f(f(.20+f(.45*ef[index]))+f(.35*es[index])):f(.20);
        let a=f(f(sigmoid(f(dot/Math.sqrt(m))))*coupling);
        a=f(f(a*f(1-block[p]))*f(1-block[j]));
        if(e>=4)a=f(a*delay[((e-4)*10+9)*n+p]);
        weights[index]=a;
        if(e<4)localSum=f(localSum+a);else distantSum=f(distantSum+a);
      }
      for(let e=0;e<4;e++)lw[e*n+p]=weights[e*n+p]/Math.max(1,localSum);
      for(let e=0;e<2;e++)rw[e*n+p]=c.long_range?weights[(e+4)*n+p]/Math.max(1,distantSum):0;
    }
    const crossA=this.affine(x.subarray(4*n,8*n),4,4,'fast_factor');
    const crossB=this.affine(x.subarray(12*n,16*n),4,4,'slow_factor');
    const features=new Float32Array(140*n);
    features.set(x,0);
    for(let p=0;p<n;p++) {
      for(let ch=0;ch<24;ch++) {
        const a=neighbors.map(map=>map[p]<0?0:x[ch*n+map[p]]);
        const mean=f(f(f(f(a[0]+a[1])+a[2])+a[3])*.25);
        features[(24+ch)*n+p]=mean;
        features[(48+ch)*n+p]=f(f(a[3]-a[2])*.5);
        features[(72+ch)*n+p]=f(f(a[1]-a[0])*.5);
        features[(96+ch)*n+p]=f(mean-x[ch*n+p]);
      }
      for(let z=0;z<m;z++) {
        let local=0,distant=0;
        for(let e=0;e<4;e++) {
          const j=neighbors[e][p];if(j>=0)local=f(local+f(lw[e*n+p]*value[z*n+j]));
        }
        for(let e=0;e<2;e++)distant=f(distant+f(rw[e*n+p]*delay[(e*10+z)*n+p]));
        features[(120+z)*n+p]=local;
        features[(128+z)*n+p]=f(c.long_gain*distant);
      }
      for(let z=0;z<4;z++) features[(136+z)*n+p]=c.interaction?f(f(Math.tanh(crossA[z*n+p]))*f(Math.tanh(crossB[z*n+p]))):0;
    }
    const hidden=this.affine(features,140,c.hidden,'update.0');
    for(let i=0;i<hidden.length;i++)hidden[i]=Math.max(0,hidden[i]);
    const delta=this.affine(hidden,c.hidden,20,'update.2');
    const cells=new Float32Array(24*n);
    for(let p=0;p<n;p++) {
      let localDv=0,distantDv=0;
      for(let e=0;e<4;e++) {const j=neighbors[e][p];localDv=f(localDv+f(lw[e*n+p]*f((j<0?0:x[4*n+j])-x[4*n+p])));}
      for(let e=0;e<2;e++)distantDv=f(distantDv+f(rw[e*n+p]*f(delay[(e*10+m)*n+p]-x[4*n+p])));
      const feedback=c.consolidation?f(f(.015*x[21*n+p])*f(1-clip(x[15*n+p],0,1))):0;
      const extra=f(f(f(f(.12*localDv)+f(.025*distantDv))+f(.15*current[p]))+feedback);
      for(let ch=0;ch<20;ch++) {
        const i=ch*n+p;
        let d=f(f(Math.tanh(delta[i]))*this.weights.rates[ch]);
        d=f(d+(ch===4?extra:0)); d=f(d-f(.002*x[i]));
        cells[i]=clip(f(x[i]+f(fire[p]*d)),ch<4||ch===15?0:-1,1);
      }
      const trace=f(x[20*n+p]+f(f(.04*fire[p])*f(x[4*n+p]-x[20*n+p])));
      const concordance=clip(f(x[4*n+p]*x[12*n+p]),0,1);
      const pref=f(x[21*n+p]+f(f(.003*fire[p])*f(f(concordance*trace)-x[21*n+p])));
      cells[20*n+p]=c.consolidation?clip(trace,-1,1):0;
      cells[21*n+p]=c.consolidation?clip(pref,-1,1):0;
      cells[22*n+p]=x[22*n+p]; cells[23*n+p]=x[23*n+p];
    }
    const keep=new Uint8Array(n);
    for(let p=0;p<n;p++)keep[p]=this.frontier(x,p)&&this.frontier(cells,p)?1:0;
    for(let ch=0;ch<24;ch++)for(let p=0;p<n;p++)cells[ch*n+p]*=keep[p];
    const edgeFast=new Float32Array(6*n),edgeSlow=new Float32Array(6*n),delayed=new Float32Array(20*n);
    for(let p=0;p<n;p++) {
      let sumFast=0;for(let e=0;e<6;e++)sumFast=f(sumFast+ef[e*n+p]);
      for(let e=0;e<6;e++) {
        const j=neighbors[e][p], index=e*n+p;
        const vj=j<0?0:x[4*n+j], rj=j<0?0:x[12*n+j];
        const agreement=clip(f(1-f(Math.abs(f(vj-x[4*n+p]))*.5)),0,1);
        const structural=clip(f(1-f(Math.abs(f(rj-x[12*n+p]))*.5)),0,1);
        const eligible=f(f(f(agreement*structural)*Math.abs(x[4*n+p]))*f(1-x[15*n+p]));
        const transfer=c.consolidation?f(.003*ef[index]):0;
        const headroom=Math.max(f(f(1-ef[index])-es[index]),0);
        const growth=f(f(f(.06*eligible)*headroom)/f(1+sumFast));
        const nf=clip(f(f(f(ef[index]+growth)-f(.02*ef[index]))-transfer),0,1);
        const ns=clip(f(f(es[index]+transfer)-f(.0005*es[index])),0,1);
        const scale=Math.max(1,f(nf+ns));
        const valid=cells[p]>.1&&j>=0&&cells[j]>.1&&(e<4||c.long_range);
        if(valid&&c.plasticity) {edgeFast[index]=nf/scale;if(c.consolidation)edgeSlow[index]=ns/scale;}
      }
      for(let e=0;e<2;e++) {
        const j=neighbors[e+4][p];
        if(j<0||cells[p]<=.1||!c.long_range)continue;
        for(let z=0;z<m;z++)delayed[(e*10+z)*n+p]=value[z*n+j];
        delayed[(e*10+m)*n+p]=x[4*n+j];
        delayed[(e*10+m+1)*n+p]=x[j]>.1?1:0;
      }
    }
    return {cells,edgeFast,edgeSlow,delayed};
  }

  damage(state, keep) {
    const s=cloneState(state),n=this.n;
    for(let p=0;p<n;p++) {
      for(let ch=0;ch<24;ch++)s.cells[ch*n+p]*=keep[p];
      for(let e=0;e<6;e++) {
        const j=this.neighbors[e][p], valid=j>=0?keep[p]*keep[j]:0;
        s.edgeFast[e*n+p]*=valid;s.edgeSlow[e*n+p]*=valid;
        if(e>=4)for(let ch=0;ch<10;ch++)s.delayed[((e-4)*10+ch)*n+p]*=valid;
      }
    }return s;
  }
}
