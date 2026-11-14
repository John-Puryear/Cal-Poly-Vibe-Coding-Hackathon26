const $=id=>document.getElementById(id);
const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
let ready=false,playing=false,busy=false,latest=null,hover=null,error=null,twinLayer='voltage',placedAt=null;
const offscreen=document.createElement('canvas');
const contexts={anatomy:$('anatomy').getContext('2d'),voltage:$('voltage').getContext('2d'),twin:$('twin').getContext('2d')};

function setPlay(value){playing=value;$('play').innerHTML=value?'<span aria-hidden="true">Ⅱ</span> Pause':'<span aria-hidden="true">▶</span> Grow tissue';}
function controls(){for(const id of ['play','advance','reset','export','center-input','specimen']){if($(id))$(id).disabled=!ready||busy;}}
function number(value){return(value>=0?'+':'')+value.toFixed(3);}
// Mean absolute difference of the four anatomy channels (body, head, eye, tail) over tissue present in either copy.
function shapeDifference({cells,baseline,width,height}){
 const n=width*height;let occupied=0,diff=0;
 for(let p=0;p<n;p++)if(cells[p]>.1||baseline[p]>.1){occupied++;for(let c=0;c<4;c++)diff+=Math.abs(cells[c*n+p]-baseline[c*n+p])/4;}
 return occupied?diff/occupied:0;
}
function paint(){
 if(!latest)return;
 const {width:w,height:h}=latest,n=w*h;
 offscreen.width=w;offscreen.height=h;
 const off=offscreen.getContext('2d'),image=off.createImageData(w,h);
 // Panels A and B show the treated tissue; panel C shows the untreated twin in the layer its switch selects.
 for(const panel of ['anatomy','voltage','twin']){
  const treated=panel!=='twin',layer=treated?panel:twinLayer,cells=treated?latest.cells:latest.baseline;
  const visible=!treated||$(layer+'-toggle').checked;
  if(treated)$(layer+'-view').querySelector('.layer-off').hidden=visible;
  const ctx=contexts[panel],canvas=ctx.canvas;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!visible)continue;
  for(let p=0;p<n;p++){
   const live=Math.max(0,Math.min(1,cells[p])),i=p*4;
   let r,g,b;
   if(layer==='anatomy'){
    const head=cells[n+p],eye=Math.max(0,Math.min(1,cells[2*n+p])),tail=cells[3*n+p];
    r=(.50+.25*head)*(1-eye*.95);g=(.65+.18*head-.25*tail)*(1-eye*.95);b=(.40+.20*tail)*(1-eye*.95);
    r*=255;g*=255;b*=255;
   }else{
    const v=Math.max(-1,Math.min(1,cells[4*n+p])),a=Math.abs(v),end=v<0?[76,158,240]:[232,178,111];
    r=37+(end[0]-37)*a;g=50+(end[1]-50)*a;b=64+(end[2]-64)*a;
   }
   image.data[i]=r;image.data[i+1]=g;image.data[i+2]=b;image.data[i+3]=live>.05?live*255:0;
  }
  off.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(offscreen,0,0,canvas.width,canvas.height);
  if($('grid-toggle').checked){
   ctx.strokeStyle='rgba(11,16,21,.25)';ctx.lineWidth=.6;ctx.beginPath();
   for(let x=1;x<w;x++){ctx.moveTo(x*canvas.width/w,0);ctx.lineTo(x*canvas.width/w,canvas.height);}
   for(let y=1;y<h;y++){ctx.moveTo(0,y*canvas.height/h);ctx.lineTo(canvas.width,y*canvas.height/h);}ctx.stroke();
  }
  if(treated){
   for(const event of latest.active){ctx.strokeStyle='rgba(232,238,236,.75)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();ctx.arc((event.x+.5)*canvas.width/w,(event.y+.5)*canvas.height/h,event.radius*canvas.width/w,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);}
   if(hover?.layer===layer){ctx.strokeStyle='rgba(129,201,190,.8)';ctx.lineWidth=1;ctx.beginPath();ctx.arc((hover.x+.5)*canvas.width/w,(hover.y+.5)*canvas.height/h,Number($('radius').value)*canvas.width/w,0,Math.PI*2);ctx.stroke();}
  }
 }
}
// Guided demo: grown once the tissue has developed, signalled once an input is placed, compared once it has run a few steps.
function markSteps(step){
 const done=[step>0,placedAt!==null,placedAt!==null&&step>=placedAt+12];
 const current=done.indexOf(false);
 done.forEach((value,i)=>{const item=$('demo-step-'+(i+1));item.classList.toggle('done',value);
  if(i===current)item.setAttribute('aria-current','step');else item.removeAttribute('aria-current');});
}
function updateFrame(data){latest=data;markSteps(data.step);$('step-value').textContent=String(data.step).padStart(3,'0');
 $('mean-voltage').textContent=number(data.metrics.meanVoltage);
 $('voltage-diff').textContent=data.metrics.voltageDifference.toFixed(4);
 $('shape-diff').textContent=shapeDifference(data).toFixed(4);
 const active=data.active.length;
 $('run-note').textContent=active?`${active} local input${active>1?'s':''} active · ${Math.max(...data.active.map(e=>e.remaining))} steps remaining`:
  data.step===0?'Starting outline ready. Press Grow tissue.':playing?'Running':'Paused';
 $('fact-grid').textContent=$('grid-corner').textContent=`${data.width} × ${data.height}`;
 paint();
}
worker.addEventListener('message',({data})=>{
 if(data.type==='ready'){
  ready=true;controls();$('model-status').innerHTML='<span class="status-dot"></span> Live model, running in your browser';
  // Model size facts come from the loaded model, so they stay correct after retraining.
  $('fact-parameters').textContent=data.parameters.toLocaleString('en-US');$('fact-channels').textContent=data.channels;
 }else if(data.type==='frame')updateFrame(data);
 else if(data.type==='paused'){busy=false;setPlay(false);controls();if(latest)updateFrame(latest);}
 else if(data.type==='error'){
  error=data.message;ready=false;busy=false;setPlay(false);controls();$('model-status').textContent='Model unavailable';
  $('run-note').textContent='Unable to run: '+data.message;$('run-note').setAttribute('role','alert');console.error(data.message);
 }else if(data.type==='export'){
  const blob=new Blob([JSON.stringify(data.record,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=`kaia-${data.record.specimen}-step-${data.record.step}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
});
worker.addEventListener('error',event=>{ready=false;controls();$('model-status').textContent='Model unavailable';$('run-note').textContent=event.message??'Worker failed to start';});
$('play').addEventListener('click',()=>{if(!ready||busy)return;setPlay(!playing);worker.postMessage({type:playing?'play':'pause'});if(!playing&&latest)updateFrame(latest);});
$('advance').addEventListener('click',()=>{if(!ready||busy)return;busy=true;setPlay(false);controls();$('run-note').textContent='Skipping ahead 48 steps…';worker.postMessage({type:'step',count:48});});
function reset(){setPlay(false);busy=false;placedAt=null;controls();worker.postMessage({type:'reset',specimen:$('specimen').value});}
$('reset').addEventListener('click',reset);$('specimen').addEventListener('change',reset);
for(const id of ['anatomy-toggle','voltage-toggle','grid-toggle'])$(id).addEventListener('change',()=>{paint();if(latest)updateFrame(latest);});
for(const button of document.querySelectorAll('.twin-option'))button.addEventListener('click',()=>{
 twinLayer=button.dataset.layer;
 for(const other of document.querySelectorAll('.twin-option'))other.setAttribute('aria-pressed',String(other===button));
 paint();
});
for(const id of ['strength','radius','duration'])$(id).addEventListener('input',()=>{
 const value=Number($(id).value);$(id+'-value').textContent=id==='strength'?(value>=0?'+':'')+value.toFixed(2):`${value} ${id==='radius'?'cells':'steps'}`;paint();
});
function intervene(x,y){if(!ready||busy||!latest)return;
 if(placedAt===null)placedAt=latest.step;
 worker.postMessage({type:'intervene',x,y,radius:Number($('radius').value),strength:Number($('strength').value),duration:Number($('duration').value)});
}
for(const layer of ['anatomy','voltage']){
 const canvas=$(layer);
 function point(event){const box=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(latest.width-1,(event.clientX-box.left)/box.width*latest.width-.5)),y:Math.max(0,Math.min(latest.height-1,(event.clientY-box.top)/box.height*latest.height-.5))};}
 canvas.addEventListener('pointermove',event=>{if(!latest)return;hover={...point(event),layer};paint();});
 canvas.addEventListener('pointerleave',()=>{hover=null;paint();});
 canvas.addEventListener('pointerdown',event=>{if(!latest||!$(layer+'-toggle').checked)return;event.preventDefault();const p=point(event);intervene(p.x,p.y);});
}
$('export').addEventListener('click',()=>worker.postMessage({type:'export'}));
const centerButton=document.createElement('button');centerButton.id='center-input';centerButton.className='quiet-button';centerButton.textContent='Place at center';centerButton.disabled=true;centerButton.addEventListener('click',()=>{if(latest)intervene(Math.round(latest.width/2),Math.round(latest.height/2));});
document.querySelector('.input-title').append(centerButton);
window.addEventListener('pagehide',()=>worker.terminate());
controls();
worker.postMessage({type:'init'});
