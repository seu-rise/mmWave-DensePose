import {api} from './api.js';

const status=document.getElementById('status'),retry=document.getElementById('retry');
let timer=null,generation=0;
function show(info,request){
  if(request!==generation)return;
  if(info.instance_running&&typeof info.url==='string'&&/^\/tensorboard\/[a-zA-Z0-9_-]+\/$/.test(info.url)){
    location.replace(info.url);return;
  }
  if(info.instance_starting||info.status==='Starting'){
    status.textContent='Loading all training logs…';
    timer=setTimeout(()=>poll(request),700);return;
  }
  fail(new Error(info.status==='events_unavailable'?'No TensorBoard event files are available.':'TensorBoard could not start.'),request);
}
function fail(error,request){
  if(request!==generation)return;
  status.textContent=error.message||'Unable to open TensorBoard.';retry.hidden=false;
}
async function poll(request){
  try{show(await api('/api/logs/tensorboard'),request)}catch(error){fail(error,request)}
}
async function start(){
  clearTimeout(timer);const request=++generation;retry.hidden=true;
  status.textContent='Starting TensorBoard for all training logs…';
  try{show(await api('/api/logs/tensorboard',{method:'POST',body:{action:'start'}}),request)}catch(error){fail(error,request)}
}
retry.addEventListener('click',start);
window.addEventListener('pagehide',()=>{generation++;clearTimeout(timer)});
start();
