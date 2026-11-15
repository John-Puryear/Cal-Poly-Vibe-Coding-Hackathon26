const $=id=>document.getElementById(id), worker=new Worker('./protocol-worker.js',{type:'module'});
let latest=null,target=null,ready=false,cutPosteriorVoltage=null,cutLabel='cut';
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const mix=(a,b,t)=>a.map((v,i)=>Math.round(v*(1-t)+b[i]*t));
function draw(canvas,cells,w,h,mode){
  if(!cells)return;
  const n=w*h,mini=document.createElement('canvas');mini.width=w;mini.height=h;
  const ctx=mini.getContext('2d'),im=ctx.createImageData(w,h),choice=$('gene-channel').value;
  for(let p=0;p<n;p++){
    const body=clamp(cells[p]),i=4*p;
    let c=[15,23,30];
    if(body>.1){
      if(mode==='anatomy'){
        c=[116,160,111];c=mix(c,[213,203,107],clamp(cells[n+p])*.86);
        c=mix(c,[126,95,177],clamp(cells[3*n+p])*.85);
        c=mix(c,[18,27,24],clamp(cells[2*n+p])*.98);
      }else if(mode==='voltage'){
        const v=clamp(cells[4*n+p],-1,1);
        // One fixed, smooth contrast curve for every frame. Numeric readouts remain raw.
        const displayStrength=Math.asinh(3*Math.abs(v))/Math.asinh(3);
        c=mix([107,121,126],v>=0?[242,67,58]:[50,100,244],displayStrength);
      }else{
        const r=clamp(cells[12*n+p],-1,1),a=clamp(cells[13*n+p],-1,1),s=clamp(cells[14*n+p]);
        if(choice==='regulatory')c=mix([45,35,55],r>=0?[233,108,176]:[91,78,188],Math.abs(r));
        else if(choice==='axis')c=mix([56,62,71],a>=0?[240,177,86]:[68,192,207],Math.abs(a));
        else if(choice==='structure')c=mix([42,53,47],[148,211,110],s);
        else c=[Math.round(76+148*clamp(r)),Math.round(67+142*s),Math.round(82+147*clamp(-a))];
      }
      c=mix([15,23,30],c,body);
    }
    im.data.set([...c,255],i);
  }
  ctx.putImageData(im,0,0);
  const out=canvas.getContext('2d');out.imageSmoothingEnabled=false;
  out.clearRect(0,0,canvas.width,canvas.height);
  const scale=Math.min(canvas.width/w,canvas.height/h),dw=w*scale,dh=h*scale;
  out.drawImage(mini,(canvas.width-dw)/2,(canvas.height-dh)/2,dw,dh);
}
function paint(){if(!latest)return;const {cells,width,height}=latest;
  draw($('anatomy-canvas'),cells,width,height,'anatomy');draw($('voltage-canvas'),cells,width,height,'voltage');
  draw($('gene-canvas'),cells,width,height,'gene');if(target)draw($('target-preview'),target,width,height,'anatomy');}
function controls(){if(!latest)return;const run=latest.running;
  $('develop').disabled=!ready||run;$('cut').disabled=!ready||run||!latest.developedReady;
  for(const b of document.querySelectorAll('[data-branch]')){
    b.disabled=!ready||run||!latest.fragmentReady;
    const selected=latest.activeBranch===b.dataset.branch;
    b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));
  }
  $('recut').disabled=!ready||run||!latest.recutReady;$('pause').disabled=!run;
  $('step-one').disabled=run||!latest.paused;
  $('resume').disabled=run||!latest.paused;$('reset').disabled=!ready;$('export').disabled=!ready;
}
function frame(f){latest=f;$('step-value').textContent=String(f.tick).padStart(4,'0');
  $('phase-name').textContent=f.phaseName;$('phase-progress').textContent=`${f.phaseTick} / ${f.phaseTotal}`;
  $('phase-fill').style.width=`${100*clamp(f.phaseTick/Math.max(1,f.phaseTotal))}%`;
  $('phase-note').textContent=f.note;
  const anterior=Number(f.metrics.anteriorVoltage),posterior=Number(f.metrics.posteriorVoltage);
  if(f.phaseName==='PRECURSOR'||f.phaseName==='DEVELOPING')cutPosteriorVoltage=null;
  if(f.phaseName==='TRUNK FRAGMENT'&&Number.isFinite(posterior)){
    cutPosteriorVoltage=posterior;cutLabel='cut';
  }else if(f.phaseName==='CLEAN RECUT'&&f.phaseTick===0&&Number.isFinite(posterior)){
    cutPosteriorVoltage=posterior;cutLabel='recut';
  }
  $('anterior-voltage').textContent=Number.isFinite(anterior)?anterior.toFixed(3):'—';
  $('posterior-voltage').textContent=Number.isFinite(posterior)?posterior.toFixed(3):'—';
  const delta=posterior-cutPosteriorVoltage;
  $('posterior-voltage-change').textContent=Number.isFinite(delta)&&cutPosteriorVoltage!==null?
    `Δ since ${cutLabel} ${delta>=0?'+':''}${delta.toFixed(3)}`:'Raw signed voltage';
  const blockActive=f.phaseName==='TREATMENT';
  $('treatment-indicator').textContent=blockActive?'Coupling block active':'Coupling block inactive';
  $('treatment-indicator').classList.toggle('active',blockActive);
  $('posterior-memory').textContent=f.metrics.posteriorMemory.toFixed(3);
  $('posterior-head').textContent=f.metrics.posteriorHead.toFixed(3);paint();controls();}
worker.onmessage=({data:d})=>{
  if(d.type==='ready'){ready=true;target=new Float32Array(d.target);
    $('stat-parameters').textContent=d.parameters.toLocaleString();
    $('model-status').innerHTML='<span class="status-dot"></span> Browser protocol ready';
    worker.postMessage({type:'speed',speed:$('playback-speed').value});}
  else if(d.type==='frame')frame(d);
  else if(d.type==='error'){$('model-status').innerHTML='<span class="status-dot amber"></span> Simulation error';
    $('phase-note').textContent=d.message;console.error(d.message);}
  else if(d.type==='export'){const blob=new Blob([JSON.stringify(d.record)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='bdh-protocol-session.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
};
$('develop').onclick=()=>worker.postMessage({type:'develop'});
$('cut').onclick=()=>worker.postMessage({type:'cut'});
for(const b of document.querySelectorAll('[data-branch]'))b.onclick=()=>worker.postMessage({type:'branch',branch:b.dataset.branch});
$('recut').onclick=()=>worker.postMessage({type:'recut'});
for(const name of ['pause','resume','reset','export'])$(name).onclick=()=>worker.postMessage({type:name});
$('step-one').onclick=()=>worker.postMessage({type:'step-one'});
$('playback-speed').onchange=()=>worker.postMessage({type:'speed',speed:$('playback-speed').value});
$('gene-channel').onchange=paint;
window.addEventListener('pagehide',()=>worker.terminate());worker.postMessage({type:'init'});
