import {ProtocolRuntime,cloneProtocolState,makeRng} from './protocol-runtime.js';

let runtime,example,state,developed,fragment,treatedWashed,program=null,timer=null;
let tick=0,phaseName='PRECURSOR',phaseTick=0,phaseTotal=128,note='The intact precursor is ready to develop.';
let activeBranch=null,paused=false,speed='live';
const BRANCHES={sham:{dose:0,duration:0,label:'Untreated control'},brief:{dose:1,duration:24,label:'Brief broad block'},sustained:{dose:1,duration:96,label:'Sustained broad block'},half:{dose:.5,duration:96,label:'Half-dose broad block'}};
const N=32*32,keep=new Float32Array(N),middle=new Float32Array(N);
for(let y=0;y<32;y++)for(let x=0;x<32;x++){const p=y*32+x;keep[p]=x>=10&&x<=23?1:0;middle[p]=x>=12&&x<=23?1:0;}

function stop(){if(timer!==null){clearTimeout(timer);timer=null;}paused=!!program;}
function mask(rng){const fire=new Float32Array(N);for(let i=0;i<N;i++)fire[i]=rng()<.5?1:0;return fire;}
function metrics(){let live=0,voltage=0,memory=0,posterior=0,posteriorCount=0;
  for(let y=0;y<32;y++)for(let x=0;x<32;x++){
    const p=y*32+x;if(state.cells[p]>.1){live++;voltage+=state.cells[4*N+p];
      if(x>=20&&x<=23){memory+=state.induction[p];posterior=Math.max(posterior,state.cells[N+p]);posteriorCount++;}}
  }
  return {meanVoltage:live?voltage/live:0,posteriorMemory:posteriorCount?memory/posteriorCount:0,posteriorHead:posterior};
}
function frame(){postMessage({type:'frame',cells:state.cells.slice(),width:32,height:32,tick,phaseName,phaseTick,phaseTotal,note,
  running:timer!==null,paused,developedReady:!!developed,fragmentReady:!!fragment,recutReady:!!treatedWashed,
  activeBranch,metrics:metrics()});}
function reset(){stop();state=runtime.initial(new Float32Array(example.seed));developed=null;fragment=null;treatedWashed=null;
  program=null;tick=0;phaseName='PRECURSOR';phaseTick=0;phaseTotal=128;note='The intact precursor is ready to develop.';activeBranch=null;paused=false;frame();}
function configure(kind,branch=null){stop();paused=false;
  if(kind==='develop'){
    state=runtime.initial(new Float32Array(example.seed));developed=null;fragment=null;treatedWashed=null;activeBranch=null;
    tick=0;phaseName='DEVELOPING';phaseTick=0;phaseTotal=128;note='Normal tissue is developing from the intact precursor.';
    program={kind,i:0,total:128,rng:makeRng(24000)};
  }else if(kind==='branch'){
    if(!fragment||!BRANCHES[branch])throw Error('Create a trunk fragment before starting a treatment.');
    state=cloneProtocolState(fragment);activeBranch=branch;const b=BRANCHES[branch];
    tick=128;phaseTick=0;phaseTotal=b.duration||608;phaseName=b.duration?'TREATMENT':'UNTREATED REGENERATION';
    note=`${b.label} from the saved trunk fragment.`;
    program={kind,branch,i:0,total:608,rng:makeRng(41000)};
  }else if(kind==='recut'){
    if(!treatedWashed)throw Error('Run sustained broad block through washout before the clean recut.');
    state=runtime.damage(cloneProtocolState(treatedWashed),keep);activeBranch='recut';tick=736;
    phaseName='CLEAN RECUT';phaseTick=0;phaseTotal=512;
    note='Communication is restored. The treated trunk was cut again and is regenerating without block.';
    program={kind,i:0,total:512,rng:makeRng(71000)};
  }
  run();
}
function execute(){const p=program;if(!p)return;
  let control=null;
  if(p.kind==='branch'){
    const b=BRANCHES[p.branch];
    if(p.i<b.duration&&b.dose){control=new Float32Array(4*N);for(let i=0;i<N;i++)control[N+i]=middle[i]*b.dose;}
  }
  state=runtime.step(state,control,mask(p.rng));p.i++;
  tick=p.kind==='develop'?p.i:p.kind==='branch'?128+p.i:736+p.i;
  if(p.kind==='branch'){
    const b=BRANCHES[p.branch];
    if(p.i<b.duration){phaseName='TREATMENT';phaseTick=p.i;phaseTotal=b.duration;
      note=`${b.label}: reversible middle-band coupling block is active.`;}
    else{phaseName=p.branch==='sham'?'UNTREATED REGENERATION':'WASHOUT + REGENERATION';
      phaseTick=p.i-b.duration;phaseTotal=608-b.duration;
      note=p.branch==='sham'?'The untreated trunk is regenerating with normal communication.':
        `${b.label} is over; coupling has recovered and regeneration continues.`;}
  }else if(p.kind==='develop'){phaseTick=p.i;}
  else{phaseTick=p.i;}
  if(p.i===p.total){
    if(p.kind==='develop'){developed=cloneProtocolState(state);phaseName='DEVELOPED';phaseTick=128;phaseTotal=128;
      note='The intact synthetic planarian is developed. Make a trunk fragment to test regeneration.';}
    else if(p.kind==='branch'){
      phaseName='REGENERATION COMPLETE';phaseTick=608;phaseTotal=608;
      note=`${BRANCHES[p.branch].label} complete. Run another condition from the same saved trunk fragment.`;
      if(p.branch==='sustained')treatedWashed=cloneProtocolState(state);
    }else{phaseName='RECUT REGENERATION COMPLETE';phaseTick=512;phaseTotal=512;
      note='Clean recut complete. Inspect the posterior head field and compare with the normal control.';}
    program=null;paused=false;timer=null;
  }
}
function run(){if(!program)return;paused=false;
  const loop=()=>{try{
    timer=null;for(let j=0;j<(speed==='fast'?4:1)&&program;j++)execute();
    if(program){timer=setTimeout(loop,speed==='observe'?125:speed==='live'?25:0);}frame();
  }catch(e){stop();postMessage({type:'error',message:e.message});}};
  timer=setTimeout(loop,speed==='observe'?125:speed==='live'?25:0);frame();
}
self.onmessage=async({data:d})=>{try{
  if(d.type==='init'){
    const [portable,seed]=await Promise.all([
      fetch('./protocol/weights.json').then(r=>{if(!r.ok)throw Error('Protocol weights unavailable');return r.json();}),
      fetch('./protocol/planaria-seed.json').then(r=>{if(!r.ok)throw Error('Protocol seed unavailable');return r.json();})]);
    if(seed.width!==32||seed.height!==32)throw Error('Expected a 32 × 32 planarian seed.');
    example=seed;runtime=new ProtocolRuntime(portable,32,32);
    postMessage({type:'ready',parameters:portable.metadata.parameter_count,target:seed.target});reset();
  }else if(d.type==='reset')reset();
  else if(d.type==='speed'){
    if(!['observe','live','fast'].includes(d.speed))throw Error('Unknown playback speed.');
    speed=d.speed;
  }
  else if(d.type==='develop')configure('develop');
  else if(d.type==='cut'){
    if(!developed)throw Error('Develop the normal planarian before cutting.');
    stop();program=null;treatedWashed=null;state=runtime.damage(cloneProtocolState(developed),keep);fragment=cloneProtocolState(state);
    tick=128;phaseName='TRUNK FRAGMENT';phaseTick=0;phaseTotal=608;
    note='Head and tail were removed. Choose a regeneration condition; each run uses this saved fragment.';activeBranch=null;paused=false;frame();
  }else if(d.type==='branch')configure('branch',d.branch);
  else if(d.type==='recut')configure('recut');
  else if(d.type==='pause'){if(program){stop();frame();}}
  else if(d.type==='step-one'){if(program){stop();execute();frame();}}
  else if(d.type==='resume'){if(program)run();}
  else if(d.type==='export')postMessage({type:'export',record:{format:'bdh-protocol-browser-session-v1',checkpoint_sha256:runtime.portable?.checkpoint_sha256??null,
    stage:phaseName,tick,branch:activeBranch,synthetic:true,clock_calibrated:false,target_used_at_inference:false,
    state:{cells:Array.from(state.cells),edgeFast:Array.from(state.edgeFast),edgeSlow:Array.from(state.edgeSlow),delayed:Array.from(state.delayed),induction:Array.from(state.induction)}}});
}catch(e){postMessage({type:'error',message:e.message});}};
