import {TissueRuntime,decodeWeights,initialState,cloneState,makeRng} from './runtime.js';
let manifest,weights,examples,provenance,runtime,state,baseline,rng,step=0,timer=null,events=[],specimen='face-0';
const seed=12345;
function decode(s){const raw=atob(s),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Float32Array(bytes.buffer);}
function stop(){if(timer!==null){clearTimeout(timer);timer=null;}}
function frame(){
  const e=examples[specimen],n=e.width*e.height;let occupied=0,mean=0,diff=0;
  for(let p=0;p<n;p++)if(state.cells[p]>.1){occupied++;mean+=state.cells[4*n+p];diff+=Math.abs(state.cells[4*n+p]-baseline.cells[4*n+p]);}
  postMessage({type:'frame',cells:state.cells.slice(),baseline:baseline.cells.slice(),width:e.width,height:e.height,step,
    active:events.filter(e=>e.end>step).map(e=>({...e,remaining:e.end-step})),
    metrics:{meanVoltage:occupied?mean/occupied:0,voltageDifference:occupied?diff/occupied:0},specimen});
}
function reset(id=specimen){stop();specimen=id;const e=examples[id];runtime=new TissueRuntime(manifest,weights,e.width,e.height);
 state=initialState(decode(e.cells),e.width,e.height);baseline=cloneState(state);rng=makeRng(seed);step=0;events=[];frame();}
function advance(count){
 const n=runtime.n;
 for(let tick=0;tick<count;tick++){
  const fire=new Float32Array(n);for(let p=0;p<n;p++)fire[p]=rng()<manifest.config.fire_rate?1:0;
  const control=new Float32Array(2*n);
  for(const event of events)if(event.start<=step&&event.end>step){
   for(let y=0;y<runtime.height;y++)for(let x=0;x<runtime.width;x++)if((x-event.x)**2+(y-event.y)**2<=event.radius**2){
    const p=y*runtime.width+x;control[p]=Math.max(-1,Math.min(1,control[p]+event.strength));control[n+p]=event.block;
   }
  }
  state=runtime.step(state,control,fire);baseline=runtime.step(baseline,null,fire);step++;
 }
 frame();
}
function play(){stop();const loop=()=>{try{advance(1);timer=setTimeout(loop,0);}catch(e){stop();postMessage({type:'error',message:e.message});}};loop();}
self.onmessage=async ({data})=>{
 try{
  if(data.type==='init'){
   [manifest,examples]=await Promise.all([fetch('./model/manifest.json').then(r=>{if(!r.ok)throw Error('Model manifest unavailable');return r.json();}),fetch('./model/initial-states.json').then(r=>{if(!r.ok)throw Error('Model precursors unavailable');return r.json();})]);
   const r=await fetch('./model/weights.f32');if(!r.ok)throw Error('Model weights unavailable');weights=decodeWeights(manifest,await r.arrayBuffer());
   const p=await fetch('./model/provenance.json');if(!p.ok)throw Error('Model provenance unavailable');provenance=await p.json();
   postMessage({type:'ready',parameters:manifest.parameter_count,channels:manifest.config.channels,specimens:Object.keys(examples)});reset();
  }else if(data.type==='reset')reset(data.specimen);
  else if(data.type==='play')play();
  else if(data.type==='pause')stop();
  else if(data.type==='step'){stop();advance(Math.min(96,Math.max(1,data.count??1)));postMessage({type:'paused'});}
  else if(data.type==='intervene'){
   const x=Math.max(0,Math.min(runtime.width-1,Math.round(data.x))),y=Math.max(0,Math.min(runtime.height-1,Math.round(data.y)));
   const duration=Math.max(1,Math.min(96,Math.round(data.duration??24)));
   events.push({x,y,radius:Math.max(1,Math.min(8,data.radius??3)),strength:Math.max(-1,Math.min(1,data.strength??.8)),block:Math.max(0,Math.min(1,data.block??0)),start:step,end:step+duration});frame();
  }else if(data.type==='export'){
   postMessage({type:'export',record:{format:'morphogenesis-session-v1',model:'bdh-nca-v2',weights_sha256:provenance.weights_sha256,config:manifest.config,update_mask_generator:'mulberry32',specimen,seed,step,
    interventions:events,states:{cells:Array.from(state.cells),edgeFast:Array.from(state.edgeFast),edgeSlow:Array.from(state.edgeSlow),delayed:Array.from(state.delayed)},
    interpretation:'Synthetic model states, not measured biology. Same update-mask stream is used for the untreated reference.'}});
  }
 }catch(e){stop();postMessage({type:'error',message:e.message});}
};
