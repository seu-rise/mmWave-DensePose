/** Selection-bound official asset transfers. Status polling never opts into page loading. */
import {helpIcon,scienceDbLink} from './ui-help.js';
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const running=job=>job?.status==='running';
const belongs=(job,id)=>Boolean(id&&Array.isArray(job?.package_ids)&&job.package_ids.includes(id));

export function officialAssetId(value){
  if(typeof value?.asset_id==='string'&&value.asset_id)return value.asset_id;
  for(const id of [value?.id,value?.weight_id])if(typeof id==='string'&&/^(paper|official):/.test(id))return id.slice(id.indexOf(':')+1)||null;
  return null;
}

export function assetBytes(value){
  const bytes=Number(value);if(value==null||!Number.isFinite(bytes)||bytes<0)return '';
  if(bytes<1024)return `${bytes} B`;
  const units=['KiB','MiB','GiB','TiB'],index=Math.min(3,Math.floor(Math.log(bytes)/Math.log(1024))-1);
  return `${(bytes/1024**(index+1)).toFixed(1)} ${units[index]}`;
}

export function assetTransferProgress(job){
  const current=Number(job?.bytes_downloaded),total=Number(job?.bytes_total);
  const percent=Number.isFinite(current)&&Number.isFinite(total)&&total>0?Math.max(0,Math.min(100,current/total*100)):null;
  return {percent,label:percent===null?'':`${assetBytes(current)} / ${assetBytes(total)}`};
}

export function paperAssetStatusHtml(state,{title='Official weights',downloadLabel='Download official weights'}={}){
  const item=state.item,job=belongs(state.job,item?.package_id)?state.job:null,busy=running(job),otherJob=running(state.activeJob)&&!belongs(state.activeJob,item?.package_id);
  const labels={installed:'Installed',missing:'Not installed',partial:'Incomplete',unverified:'Verification required',unpublished:'Not published',unavailable:'Unavailable'};
  const stages={starting:'Preparing',verifying_cache:'Checking cached packages',verifying:'Verifying',downloading:'Downloading',downloaded:'Downloaded',extracting:'Extracting',installing:'Installing',installed:'Installed',cancelling:'Cancelling'};
  const terminal=item?.availability!=='installed'?({failed:'Failed',cancelled:'Cancelled',interrupted:'Interrupted'})[job?.status]:null;
  const status=state.loading?'Checking…':busy?stages[job.stage]||'Downloading':terminal||labels[item?.availability]||'Unavailable';
  let message=state.error||(item?.availability!=='installed'?job?.error:'')||'';
  if(!message&&busy)message=stages[job.stage]||'Preparing download';
  if(!message&&job?.status==='cancelled')message='Download cancelled. Verified packages can be reused.';
  if(!message&&otherJob)message='Another official asset download is running.';
  if(!message&&!state.loading&&item?.availability!=='installed')message=item?.reason||'This asset is not available for download.';
  const progress=assetTransferProgress(job);
  const action=item?.requires_repair&&item?.can_repair?'repair':'download',canTransfer=Boolean(item?.can_download||action==='repair'&&item?.can_repair)&&item?.availability!=='installed';
  const resume=Boolean(item?.can_resume||['partial','unverified'].includes(item?.availability)||['failed','cancelled','interrupted'].includes(job?.status));
  const transferLabel=action==='repair'?'Repair official asset':item?.availability==='unverified'?'Verify installed files':resume?'Continue download':downloadLabel;
  const button=busy?'<button type="button" data-paper-asset-action="cancel">Cancel</button>':canTransfer?`<button type="button" data-paper-asset-action="${action}" ${state.pending||otherJob?'disabled':''}>${escape(state.pending?'Starting…':transferLabel)}</button>`:'';
  const cacheLabel=!busy&&item?.availability!=='installed'&&(item?.cache_available||item?.complete_transfer_available)?'Cached package ready for verification':!busy&&item?.availability!=='installed'&&item?.partial_download_bytes?`${assetBytes(item.partial_download_bytes)} downloaded; transfer incomplete`:'';
  const detail=busy&&progress.label?`${message} · ${progress.label}`:[message,cacheLabel].filter(Boolean).join(' · ');
  return `<section class="paper-asset-transfer" data-loading="false" aria-label="${escape(title)}"><div class="paper-asset-summary"><span class="paper-asset-title">${escape(title)}${scienceDbLink("Open official files on ScienceDB")}${helpIcon("Install this official package and verify its files on this server.",title+" help")}</span><span class="paper-asset-state ${item?.availability==='installed'?'is-installed':''}" role="status">${escape(status)}</span></div>${detail?`<p class="paper-asset-message ${state.error||job?.error?'is-error':''}" role="${state.error||job?.error?'alert':'status'}">${escape(detail)}</p>`:''}${busy?`<progress class="paper-asset-progress" max="100" ${progress.percent===null?'':`value="${progress.percent}"`} aria-label="Official asset download progress"></progress>`:''}<div class="paper-asset-actions">${button}${!state.loading&&!busy?'<button type="button" class="paper-asset-refresh" data-paper-asset-action="refresh" aria-label="Refresh official asset status" title="Refresh status">↻</button>':''}${item?.size_bytes&&!busy?`<span>${assetBytes(item.size_bytes)}</span>`:''}</div></section>`;
}

function ensureStyles(){
  if(typeof document==='undefined'||document.querySelector('link[data-paper-assets-css]'))return;
  const link=document.createElement('link');link.rel='stylesheet';link.href=new URL('./paper-assets.css',import.meta.url).href;link.dataset.paperAssetsCss='';document.head.append(link);
}

export function createPaperAssetDownload({api,onChange=()=>{},onInstalled=()=>{},pollDelay=1200}={}){
  ensureStyles();
  const state={key:'',query:null,item:null,loading:false,pending:false,error:'',job:null,activeJob:null};
  let request=0,timer=null,disposed=false,active=true;
  const notify=()=>{if(!disposed)onChange(state)};
  const stop=()=>{clearTimeout(timer);timer=null};
  const current=(version,key)=>!disposed&&version===request&&key===state.key;
  async function installed(key,query){const expected=request+1;await refresh();if(!disposed&&request===expected&&state.key===key&&!state.error){if(state.item?.availability==='installed')await onInstalled(query);else{state.error='Installed files need verification before this official asset can be used.';notify()}}}
  function schedule(){stop();if(active&&running(state.job)&&!disposed)timer=setTimeout(poll,pollDelay)}
  async function refresh(){
    if(disposed||!state.query)return;
    const version=++request,key=state.key;stop();state.loading=true;state.error='';notify();
    try{
      const payload=await api('/api/paper-assets?'+new URLSearchParams(state.query));
      if(!current(version,key))return;
      state.item=(Array.isArray(payload.items)?payload.items:[]).find(item=>item.kind===state.query.kind&&item.asset_id===state.query.asset_id&&(!state.query.test_id||item.test_id===state.query.test_id))||null;
      state.activeJob=payload.active_job||null;
      state.job=belongs(payload.active_job,state.item?.package_id)?payload.active_job:belongs(payload.latest_job,state.item?.package_id)?payload.latest_job:null;
      if(!state.item)state.error='Official asset metadata unavailable.';
    }catch(error){if(current(version,key))state.error=error.message||'Unable to check official asset status.'}
    finally{if(current(version,key)){state.loading=false;notify();schedule()}}
  }
  function select(query){
    const key=query?JSON.stringify(query):'';
    if(key===state.key)return Promise.resolve();
    request++;stop();Object.assign(state,{key,query:query?{...query}:null,item:null,job:null,activeJob:null,error:'',pending:false,loading:false});notify();
    return query?refresh():Promise.resolve();
  }
  async function start(repair=false){
    if(state.pending||!state.item||running(state.job)||running(state.activeJob))return;
    if(!(repair?state.item.can_repair:state.item.can_download))return;
    const key=state.key,version=++request,id=state.item.package_id;stop();state.pending=true;state.error='';notify();
    try{
      const job=await api('/api/paper-assets/installations',{method:'POST',body:{package_ids:[id],...(repair?{repair:true}:{})}});
      if(!current(version,key))return;
      if(!belongs(job,id))throw Error('Download response does not match the selected official asset.');
      state.job=job;state.activeJob=running(job)?job:null;state.pending=false;notify();
      if(job.status==='completed')await installed(key,{...state.query});else schedule();
    }catch(error){if(current(version,key))state.error=error.message||'Unable to start the download.'}
    finally{if(current(version,key)){state.pending=false;notify()}}
  }
  async function poll(){
    if(disposed||!active||!running(state.job))return;
    const key=state.key,version=request,id=state.job.id;
    try{
      const job=await api('/api/paper-assets/installations/'+encodeURIComponent(id));
      if(!current(version,key)||state.job?.id!==id)return;
      state.job=job;state.activeJob=running(job)?job:null;notify();
      if(job.status==='completed'){
        await installed(key,{...state.query});
      }else schedule();
    }catch(error){if(current(version,key)){state.error=error.message||'Unable to refresh download progress.';notify();schedule()}}
  }
  async function cancel(){
    if(state.pending||!running(state.job))return;
    const key=state.key,version=request,id=state.job.id;state.pending=true;notify();
    try{
      const job=await api('/api/paper-assets/installations/'+encodeURIComponent(id)+'/cancel',{method:'POST',body:{}});
      if(!current(version,key))return;
      state.job=job;state.activeJob=running(job)?job:null;state.pending=false;
      if(job.status==='completed')await installed(key,{...state.query});else schedule();
    }catch(error){if(current(version,key))state.error=error.message||'Unable to cancel the download.'}
    finally{if(current(version,key)){state.pending=false;notify()}}
  }
  async function handleClick(event){
    const button=event.target.closest('[data-paper-asset-action]');if(!button)return false;
    const action=button.dataset.paperAssetAction;
    if(action==='refresh')await refresh();else if(action==='cancel')await cancel();else if(action==='download'||action==='repair')await start(action==='repair');
    return true;
  }
  return {state,select,refresh,start,cancel,handleClick,setActive(value){const enabled=Boolean(value);if(enabled===active)return;active=enabled;stop();if(active&&state.query)return refresh()},dispose(){disposed=true;request++;stop()}};
}
