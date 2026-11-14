import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {TissueRuntime,decodeWeights} from '../dist/runtime.js';
const root=new URL('../',import.meta.url);
const manifest=JSON.parse(fs.readFileSync(new URL('dist/model/manifest.json',root)));
const binary=fs.readFileSync(new URL('dist/model/weights.f32',root));
const weights=decodeWeights(manifest,binary.buffer.slice(binary.byteOffset,binary.byteOffset+binary.byteLength));
const fixture=JSON.parse(fs.readFileSync(new URL('tests/fixture.json',root)));
const shape=fixture.input_0.shape, h=shape.at(-2),w=shape.at(-1);
if(shape[0]!==1)throw new Error('Expected single-sample fixture');
const runtime=new TissueRuntime(manifest,weights,w,h);
let state={cells:Float32Array.from(fixture.input_0.data),edgeFast:Float32Array.from(fixture.input_1.data),
 edgeSlow:Float32Array.from(fixture.input_2.data),delayed:Float32Array.from(fixture.input_3.data)};
let error=0;const start=performance.now();
for(let tick=0;tick<manifest.fixture_ticks;tick++){
  state=runtime.step(state,Float32Array.from(fixture['control_'+tick].data),Float32Array.from(fixture['fire_'+tick].data));
  for(const [i,key]of ['cells','edgeFast','edgeSlow','delayed'].entries()) {
    const expected=fixture[`output_${tick}_${i}`].data;
    let local=0;for(let j=0;j<expected.length;j++)local=Math.max(local,Math.abs(state[key][j]-expected[j]));
    if(!Number.isFinite(local))throw new Error('Non-finite runtime state');
    error=Math.max(error,local);
    console.log(JSON.stringify({tick:tick+1,state:key,max_error:local}));
  }
}
const report={ticks:manifest.fixture_ticks,max_absolute_error:error,tolerance:1e-5,passed:error<=1e-5,
 milliseconds:performance.now()-start,grid:[w,h],all_recurrent_state_checked:true};
fs.writeFileSync(new URL('dist/model/browser-validation.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
if(!report.passed)process.exitCode=1;
