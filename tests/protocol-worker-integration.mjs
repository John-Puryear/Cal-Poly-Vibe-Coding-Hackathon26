import fs from 'node:fs/promises';
import assert from 'node:assert/strict';

let last=null,error=null,ready=null,exported=null;
globalThis.self={};
globalThis.postMessage=message=>{
  if(message.type==='frame')last=message;
  else if(message.type==='error')error=message.message;
  else if(message.type==='ready')ready=message;
  else if(message.type==='export')exported=message.record;
};
globalThis.fetch=async url=>{
  const data=await fs.readFile(new URL('../dist/'+String(url).replace(/^\.\//,''),import.meta.url));
  return {ok:true,json:async()=>JSON.parse(data.toString())};
};
await import('../dist/protocol-worker.js');
const send=async message=>{await self.onmessage({data:message});if(error)throw Error(error);return last;};
async function complete(stage){
  const start=Date.now();
  while(last?.phaseName!==stage){
    if(error)throw Error(error);
    if(Date.now()-start>120000)throw Error(`Timed out waiting for ${stage}: ${last?.phaseName}`);
    await new Promise(resolve=>setTimeout(resolve,40));
  }
  return last;
}

await send({type:'init'});
assert.equal(ready.parameters,254);
assert.equal(last.phaseName,'PRECURSOR');
await send({type:'speed',speed:'fast'});
await send({type:'develop'});
await send({type:'pause'});
assert(last.paused);
await send({type:'step-one'});
assert.equal(last.tick,1,'Single step advances exactly one model tick');
await send({type:'resume'});
await complete('DEVELOPED');
assert(last.developedReady);
await send({type:'cut'});
assert.equal(last.phaseName,'TRUNK FRAGMENT');
assert(last.fragmentReady);
const fragment=new Float32Array(last.cells);
await send({type:'branch',branch:'sham'});
assert.deepEqual(last.cells,fragment,'Untreated run must start from the saved fragment');
await complete('REGENERATION COMPLETE');
const shamPosterior=last.metrics.posteriorHead;
await send({type:'branch',branch:'sustained'});
assert.deepEqual(last.cells,fragment,'Treated run must start from the same saved fragment');
await complete('REGENERATION COMPLETE');
assert(last.recutReady);
const treatedPosterior=last.metrics.posteriorHead;
assert(treatedPosterior>shamPosterior+.2,'Sustained treatment should produce a posterior head field');
await send({type:'recut'});
await complete('RECUT REGENERATION COMPLETE');
assert(last.metrics.posteriorHead>.4,'The treated posterior head should persist after clean recut');
await send({type:'export'});
assert.equal(exported.checkpoint_sha256.length,64);
assert.equal(exported.target_used_at_inference,false);
console.log(JSON.stringify({passed:true,checks:['loads compact checkpoint','pauses and advances one tick','develops normal precursor','cuts actual developed state','reuses one fragment for sham and treatment','preserves treated head through clean recut','exports complete state'],shamPosterior,treatedPosterior,recutPosterior:last.metrics.posteriorHead}));
