import fs from 'node:fs';
import {TissueRuntime,decodeWeights,initialState} from '../dist/runtime.js';
const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root)));
const manifest=read('dist/model/manifest.json'),raw=fs.readFileSync(new URL('dist/model/weights.f32',root));
const weights=decodeWeights(manifest,raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
const examples=read('dist/model/initial-states.json'),cases=read('tests/long-fixture.json');
const results=[];
for(const test of cases){
 const {width:w,height:h}=test,n=w*h;
 const bytes=Buffer.from(examples[test.specimen].cells,'base64');
 let state=initialState(new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),w,h);
 const runtime=new TissueRuntime(manifest,weights,w,h);let maxError=0;
 for(let tick=0;tick<test.masks.length;tick++){
  const control=new Float32Array(2*n);
  if(tick>=96&&tick<120)for(let y=0;y<h;y++)for(let x=0;x<w;x++)if((x-Math.floor(w/2))**2+(y-Math.floor(h/2))**2<=9)control[y*w+x]=.8;
  state=runtime.step(state,control,Float32Array.from(test.masks[tick]));
  if(test.snapshots[String(tick+1)])for(const [i,key]of ['cells','edgeFast','edgeSlow','delayed'].entries()){
   const expected=test.snapshots[String(tick+1)][i];
   for(let j=0;j<expected.length;j++){const error=Math.abs(expected[j]-state[key][j]);if(!Number.isFinite(error))throw Error('Non-finite state');maxError=Math.max(maxError,error);}
  }
 }
 results.push({specimen:test.specimen,ticks:144,development:96,pulse:24,washout:24,max_absolute_error:maxError,tolerance:1e-5,passed:maxError<=1e-5});
}
const report={passed:results.every(x=>x.passed),reference:'Supplied independent NumPy runtime',cases:results};
fs.writeFileSync(new URL('dist/model/long-rollout-validation.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;
