/** Explicit user-operation loading; background polling/prefetch never opts in. */
const pending=new Map();
let overlay=null,labelNode=null,timer=null;

function ensureOverlay(){
  if(typeof document==='undefined'||!document.body)return null;
  if(overlay?.isConnected)return overlay;
  if(!document.getElementById('workbench-loading-css')){
    const style=document.createElement('link');style.id='workbench-loading-css';style.rel='stylesheet';style.href=new URL('./loading.css',import.meta.url).href;document.head.append(style);
  }
  overlay=document.createElement('dialog');overlay.className='workbench-loading-overlay';overlay.id='workbench-loading';overlay.setAttribute('aria-labelledby','workbench-loading-label');
  const card=document.createElement('div');card.className='workbench-loading-card';card.setAttribute('role','status');card.setAttribute('aria-live','polite');
  const spinner=document.createElement('span');spinner.className='workbench-loading-spinner';spinner.setAttribute('aria-hidden','true');
  labelNode=document.createElement('span');labelNode.id='workbench-loading-label';labelNode.textContent='Loading';card.append(spinner,labelNode);overlay.append(card);
  overlay.addEventListener('cancel',event=>{if(pending.size)event.preventDefault()});document.body.append(overlay);return overlay;
}
function update(){
  const label=Array.from(pending.values()).at(-1)||'Loading';
  if(labelNode)labelNode.textContent=label;
  if(overlay)overlay.dataset.loadingCount=String(pending.size);
  if(typeof document!=='undefined'){
    const main=document.querySelector('main');if(pending.size)main?.setAttribute('aria-busy','true');else main?.removeAttribute('aria-busy');
  }
}
function show(){
  timer=null;if(!pending.size)return;const dialog=ensureOverlay();update();
  if(dialog&&!dialog.open){try{dialog.showModal()}catch{dialog.setAttribute('open','')}}
}
function finish(token){
  pending.delete(token);update();if(pending.size)return;
  if(timer!==null){clearTimeout(timer);timer=null}
  if(overlay?.open){try{overlay.close()}catch{overlay.removeAttribute('open')}}
}
async function nextPaint(){
  if(typeof document==='undefined'||document.hidden||typeof requestAnimationFrame!=='function')return;
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
}

export async function withLoading(task,{label='Loading',delay=120}={}){
  const token=Symbol('loading');pending.set(token,label);
  try{
    update();
    if(delay<=0){if(timer!==null){clearTimeout(timer);timer=null}show()}
    else if(!overlay?.open&&timer===null)timer=setTimeout(show,delay);
    // Expensive synchronous decode work must let the visible mask paint first.
    if(delay<=0)await nextPaint();
    return await (typeof task==='function'?task():task);
  }finally{finish(token)}
}

export function bindLoadingActions(root,handlers,options={}){
  const listeners=[];
  for(const [type,handler] of Object.entries(handlers)){
    if(typeof handler!=='function')continue;
    const listener=event=>{
      const target=event.target;
      const visibleAction=type==='click'?target?.closest?.('button,[role="button"]'):target?.matches?.('select,input,textarea');
      const background=target?.closest?.('[data-loading="false"]');
      return visibleAction&&!background?withLoading(()=>handler(event),options):handler(event);
    };
    root.addEventListener(type,listener);listeners.push([type,listener]);
  }
  return ()=>{for(const [type,listener] of listeners)root.removeEventListener(type,listener)};
}
