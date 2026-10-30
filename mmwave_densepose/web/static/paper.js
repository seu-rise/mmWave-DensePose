import {tableMetricColumns,metricValue,metricPercentage} from './paper-table-metrics.js';
import {mountJointComparison} from './paper-joint-chart.js';
import {loadECharts} from './scalar-chart.js';
import {withLoading,bindLoadingActions} from './loading.js';
import {helpIcon} from './ui-help.js';
const FIGURE_URL='/paper-assets/paper-browser/catalog.js',TABLE_URL='/paper-assets/tables/paper-browser/tables.js',CAPTION_URL='/paper-assets/paper-browser/captions.json';
const FIGURES=['Scenario and framework','DDMA point cloud pipeline','Multimodal acquisition','Synchronized observations','Point count distribution','Cross-dataset comparison','19-joint definition','PAST-Net architecture','Motions 1–12','Motions 13–19 and free motion','Qualitative comparison'];
const VISIBLE_FIGURES=[...FIGURES.slice(0,8).map((title,i)=>[i+1,title,`Fig.${i+1}`]),[9,'Motions 1–19 and free motion','Fig.9–10'],[11,FIGURES[10],'Fig.11']];
const TABLES=['Data streams','Dataset statistics','Related datasets','Overall performance','Per-joint MLE','Five-seed robustness','Split protocols','Module ablation','Input features','Point count ablation'];
export function isInformationalTable(id){return Number.isInteger(Number(id))&&Number(id)>=1&&Number(id)<=3}
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const button=(label,action,cls='',extra='')=>`<button type="button" class="${cls}" data-paper-action="${action}" ${extra}>${label}</button>`;
const panel=(title,body,cls='')=>{
 const help=title==='Contents'?'Select a paper figure or benchmark table to inspect.':title.startsWith('Figure')?'Click an expandable subplot to inspect its interactive figure.':title.startsWith('Table')?'Read paper values or compare compatible complete tests from this checkout.':null;
 return `<section class="paper-panel ${cls}"><div class="paper-head"><div class="ui-title-group"><h2>${esc(title)}</h2>${help?helpIcon(help,title+" help"):''}</div></div>${body}</section>`;
};
const kv=rows=>`<dl class="paper-kv">${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
function staticResource(element,url,attribute,ready,message){
 return new Promise((resolve,reject)=>{let timer;const finish=failed=>{clearTimeout(timer);element.onload=null;element.onerror=null;if(failed||!ready()){element.remove();reject(new Error(message))}else resolve()};element.onload=()=>finish(false);element.onerror=()=>finish(true);timer=setTimeout(()=>finish(true),45000);element[attribute]=url;document.head.append(element)});
}
function script(src,name){if(globalThis[name])return Promise.resolve(globalThis[name]);const element=document.createElement('script');return staticResource(element,src,'src',()=>Boolean(globalThis[name]),'Official data unavailable').then(()=>globalThis[name])}
function styles(){if(document.getElementById('paper-css'))return Promise.resolve();const element=document.createElement('link');element.id='paper-css';element.rel='stylesheet';return staticResource(element,new URL('./paper.css',import.meta.url).href,'href',()=>true,'Official styles unavailable')}
function assetLoad(element,src){
 let finish;const promise=new Promise((resolve,reject)=>{let timer;finish=(error,paint=false)=>{clearTimeout(timer);element.removeEventListener('load',loaded);element.removeEventListener('error',failed);if(error)reject(error);else if(paint)requestAnimationFrame(()=>requestAnimationFrame(resolve));else resolve()};const loaded=()=>finish(undefined,true),failed=()=>finish(new Error('Official asset unavailable'));element.addEventListener('load',loaded,{once:true});element.addEventListener('error',failed,{once:true});timer=setTimeout(failed,45000);element.src=src});return {promise,cancel:()=>finish()};
}
export async function mountPaperPanel(root,{api}={}){
 await withLoading(styles,{delay:0});root.classList.add('paper-root','paper-workspace');
 const state={catalog:null,staticTables:null,liveTables:null,captions:null,selected:'t4',source:'reference',loading:false,error:'',request:0,active:false,previewChart:null,previewLoads:[],previewReady:Promise.resolve(),modalLoad:null,modalChart:null,modalData:null,contentsOpen:false};
 const mobilePaperQuery=window.matchMedia('(max-width:850px)');let contentsDrawer=null,contentsDrawerOpenTimer=0;
 const $=s=>root.querySelector(s);root.innerHTML='<div class="paper-empty">Loading</div>';
 const captions=api?api(CAPTION_URL):fetch(CAPTION_URL,{cache:'no-store'}).then(response=>{if(!response.ok)throw new Error('Official captions unavailable');return response.json()});
 try{[state.catalog,state.staticTables,state.captions]=await withLoading(()=>Promise.all([script(FIGURE_URL,'PAPER_FIGURE_CATALOG'),script(TABLE_URL,'PAPER_TABLE_DATA'),captions.catch(()=>({figures:{}}))]),{delay:0});render();await withLoading(()=>state.previewReady)}catch(error){root.innerHTML=`<div class="paper-empty" role="status">${esc(error.message)}</div>`}
 const unbindLoading=bindLoadingActions(root,{click:onClick});root.addEventListener('keydown',onKeyDown);window.addEventListener('message',onMessage);mobilePaperQuery.addEventListener('change',syncContentsDrawer);
 function table(){return (state.source==='measured'?state.liveTables:state.staticTables)?.tables?.[state.selected.slice(1)]}
 function missingTable(){const data=state.staticTables?.tables?.[state.selected.slice(1)]?.reference;if(!data)return null;const labels=state.selected==='t7'?2:1;return {columns:data.columns,rows:data.rows.map(row=>row.map((v,i)=>i<labels?v:null))}}
 function tableData(){return state.source==='reference'?table()?.reference:table()?.measured||missingTable()}
 function sourceMarkup(){
  const id=Number(state.selected.slice(1));
  if(state.selected[0]==='f'){
   const item=state.catalog?.[id],asset=item?.renderer==='image'?String(item.asset||'').split('/').at(-1):item?.renderer==='echarts'?'figure-05':id===9?'figure-09.json + figure-10.json':`figure-${String(id).padStart(2,'0')}.json`;
   return kv([['Renderer',item?.renderer==='image'?'Image':item?.renderer==='echarts'?'ECharts':'Three.js'],['Data','Frozen'],['Official version','v2'],...(item?.source_frame?[['Sequence',item.source_frame.file],['Frame (1-based)',item.source_frame.frame_number]]:[]),['Asset',asset]]);
  }
  const rows=table()?.provenance||[],records=new Set(rows.flatMap(row=>row.record?[row.record]:row.records||[]));
  return kv([['Source',state.source==='measured'?'Best tests':'Official'],['Table',id],['Records',state.source==='measured'?records.size:'—'],['Data',state.source==='measured'?'Local test records':`table_${String(id).padStart(2,'0')}.csv`]]);
 }
 function cell(value,row,column,metric){
  if(value===null||value===undefined||value===''||['-','–','—'].includes(String(value).trim()))return '<td class="missing-value">-</td>';
  const p=state.source==='measured'?(table()?.provenance||[]).find(x=>x.row===row&&x.column===column):null,record=p?.record||(p?.records||[]).join(' · '),formatted=typeof value==='number'?(column===0||Number(state.selected.slice(1))<=3?String(value):value.toFixed(2)):String(value),number=metricValue(value);
  const bar=metric&&number!==null,classes=[p?'measured-value':'',bar?'paper-metric-cell':'',bar&&number===metric.best?'paper-metric-best':''].filter(Boolean).join(' ');
   const content=bar?`<span class="paper-metric-wrap"><span class="paper-metric-number">${esc(formatted)}</span><svg class="paper-metric-track" viewBox="0 0 100 6" preserveAspectRatio="none" aria-hidden="true"><rect class="paper-metric-fill" width="${metricPercentage(value,metric.maximum)}" height="6" rx="3"/></svg></span>`:esc(formatted);
  return `<td${classes?` class="${classes}"`:''}${p?` data-best-record="${esc(record)}" title="${esc(record)}"`:''}${bar?` data-paper-metric="${number}" data-paper-maximum="${metric.maximum}"`:''}>${content}</td>`;
 }
 function tableMarkup(id){
  if(isInformationalTable(id))state.source='reference';
  const data=tableData(),metrics=tableMetricColumns(id,data),status=state.loading?'<span class="paper-data-state" role="status">Loading</span>':state.error?`<span class="paper-data-state paper-error" role="status">${esc(state.error)}</span>`:'',caption=state.captions?.tables?.[String(id)]?.caption||'';
  const grid=data?`<table><thead><tr>${data.columns.map((label,column)=>`<th>${esc(label)}${metrics.has(column)?`<span class="paper-metric-direction" aria-hidden="true">${metrics.get(column).higher?'↑':'↓'}</span>`:''}</th>`).join('')}</tr></thead><tbody>${data.rows.map((row,r)=>`<tr>${row.map((value,c)=>cell(value,r,c,metrics.get(c))).join('')}</tr>`).join('')}</tbody></table>`:'<div class="paper-empty">Unavailable</div>';
  const chart=id===5&&data?`<section class="paper-joint-comparison" data-paper-action="open-joint-chart" role="button" tabindex="0" aria-label="Open per-joint comparison" aria-haspopup="dialog" ${state.loading?'aria-disabled="true"':''}><div class="paper-joint-comparison-head"><h3>Per-joint MLE</h3></div><div data-paper-joint-chart></div></section>`:'';
  return `<figure class="paper-figure-block paper-table-block${id===5?' paper-joint-table-block':''}">${panel(`Table ${id} · ${TABLES[id-1]}`,`<div class="paper-canvas table-canvas"><div class="paper-toggle"><button type="button" data-paper-source="reference" class="${state.source==='reference'?'active':''}">Official</button><button type="button" data-paper-source="measured" class="${state.source==='measured'?'active':''}" ${isInformationalTable(id)?'disabled aria-disabled="true"':''}>My results</button>${helpIcon('Official shows paper values; My results uses compatible complete local evaluations.','Table data source help')}${status}</div><div class="paper-table-wrap paper-reference-table" data-paper-data>${grid}</div>${chart}</div>`)}<figcaption class="paper-figure-caption" data-paper-table-caption aria-label="Table ${id} caption">${caption?`<strong>Table ${id}.</strong> ${esc(caption)}`:''}</figcaption></figure>`;
 }
 function disposePreview(){state.previewChart?.dispose();state.previewChart=null;for(const task of state.previewLoads)task.cancel();state.previewLoads=[];state.previewReady=Promise.resolve()}
 function mountPreview(){
  const waiting=[];state.previewLoads=[...root.querySelectorAll('[data-asset-src]')].map(element=>assetLoad(element,element.dataset.assetSrc));waiting.push(...state.previewLoads.map(task=>task.promise));
  const host=$('[data-paper-joint-chart]');if(host){state.previewChart=mountJointComparison(host,tableData());state.previewChart.setActive(state.active);waiting.push(loadECharts())}
  state.previewReady=Promise.all(waiting);state.previewReady.catch(()=>{});
 }
 async function awaitPreview(selected=state.selected){try{await state.previewReady}catch(error){if(selected!==state.selected)return;state.error=error.message||'Official asset unavailable';const host=$('[data-paper-figure-stage]')||$('[data-paper-joint-chart]');if(host)host.innerHTML=`<div class="paper-empty" role="alert">${esc(state.error)}</div>`}}
 function contentMarkup(){const id=Number(state.selected.slice(1));if(state.selected[0]==='t')return tableMarkup(id);const item=state.catalog?.[id];let content;if(item?.status==='metadata_missing')content=`<div class="paper-static-strip">${(item.fallback||[]).map(p=>`<figure><img data-asset-src="/paper-assets/paper-browser/${esc(p.asset)}" alt="${esc(p.label)}"><figcaption>${esc(p.label)}</figcaption></figure>`).join('')}</div>`;else if(item?.renderer==='image')content=`<div class="paper-image-view"><img data-asset-src="/paper-assets/paper-browser/${esc(item.asset)}" alt="Figure ${id}"></div>`;else if(id===5)content='<iframe class="paper-figure-frame echarts-frame" title="Figure 5" data-asset-src="/paper-assets/figure-05/index.html?mode=preview"></iframe>';else content=`<iframe class="paper-figure-frame" title="Figure ${id}" data-asset-src="/paper-assets/paper-browser/index.html?figure=${id}${id===9?'&include=10':''}&mode=preview"></iframe>`;const caption=state.captions?.figures?.[String(id)]?.caption||'';return `<figure class="paper-figure-block">${panel(`Figure ${id===9?'9–10':id} · ${id===9?'Motions 1–19 and free motion':FIGURES[id-1]}`,`<div class="paper-canvas figure-canvas paper-figure-stage" data-paper-figure-stage data-figure="${id}" data-renderer="${esc(item?.renderer||'image')}">${content}</div>`)}<figcaption class="paper-figure-caption" data-paper-caption aria-label="Figure ${id===9?'9–10':id} caption">${caption?`<strong>Fig. ${id===9?'9–10':id}.</strong> ${esc(caption)}`:''}</figcaption></figure>`}
 function disposeContentsDrawer(){clearTimeout(contentsDrawerOpenTimer);contentsDrawerOpenTimer=0;const current=contentsDrawer;contentsDrawer=null;if(current?.dialog.open)current.dialog.close()}
 function resizePaperPreview(){requestAnimationFrame(()=>{state.previewChart?.setActive(state.active);for(const frame of root.querySelectorAll('.paper-figure-frame')){try{frame.contentWindow?.dispatchEvent(new Event('resize'))}catch(_error){}}})}
 function setContentsDrawerOpen(open){
  clearTimeout(contentsDrawerOpenTimer);contentsDrawerOpenTimer=0;
  if(!contentsDrawer||!mobilePaperQuery.matches)return;state.contentsOpen=Boolean(open);contentsDrawer.toggle.setAttribute('aria-expanded',String(state.contentsOpen));
  if(state.contentsOpen){if(!contentsDrawer.dialog.open)contentsDrawer.dialog.showModal()}else if(contentsDrawer.dialog.open)contentsDrawer.dialog.close();resizePaperPreview();
 }
 function scheduleContentsDrawerOpen(){clearTimeout(contentsDrawerOpenTimer);contentsDrawerOpenTimer=setTimeout(()=>{contentsDrawerOpenTimer=0;if(state.active&&mobilePaperQuery.matches)setContentsDrawerOpen(true)},350)}
 function syncContentsDrawer(){
  const current=contentsDrawer;if(!current)return;
  if(mobilePaperQuery.matches){current.body.append(current.contents);if(state.contentsOpen&&!current.dialog.open)current.dialog.showModal()}
  else{clearTimeout(contentsDrawerOpenTimer);contentsDrawerOpenTimer=0;state.contentsOpen=false;if(current.dialog.open)current.dialog.close();current.home.append(current.contents)}
  current.toggle.setAttribute('aria-expanded',String(state.contentsOpen));resizePaperPreview();
 }
 function mountContentsDrawer(){
  const contents=$('.paper-grid>.paper-panel');if(!contents)return;contents.classList.add('paper-contents-panel');
  const home=document.createElement('div');home.className='paper-contents-home';contents.before(home);home.append(contents);
  const toggle=document.createElement('button');toggle.type='button';toggle.className='paper-mobile-drawer-toggle';toggle.dataset.paperAction='contents-open';toggle.setAttribute('aria-label','Open figure and table contents');toggle.setAttribute('aria-haspopup','dialog');toggle.setAttribute('aria-expanded',String(state.contentsOpen));toggle.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>';
  const dialog=document.createElement('dialog');dialog.className='paper-mobile-drawer';dialog.innerHTML='<div class="paper-mobile-drawer-head"><h2>Contents</h2><button type="button" data-paper-action="contents-close" aria-label="Close contents">×</button></div><div class="paper-mobile-drawer-body"></div>';root.append(toggle,dialog);
  const drawerId=(root.id||'paper')+'-mobile-contents';dialog.id=drawerId;dialog.setAttribute('aria-label','Figure and table contents');toggle.setAttribute('aria-controls',drawerId);
  const body=dialog.querySelector('.paper-mobile-drawer-body');contentsDrawer={dialog,toggle,body,home,contents};
  dialog.addEventListener('close',()=>{if(contentsDrawer?.dialog!==dialog)return;state.contentsOpen=false;toggle.setAttribute('aria-expanded','false');resizePaperPreview()});
  dialog.addEventListener('click',event=>{if(event.target!==dialog)return;const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)setContentsDrawerOpen(false)});
  syncContentsDrawer();
 }
 function render(){disposeContentsDrawer();if(state.selected[0]==='t'&&isInformationalTable(state.selected.slice(1)))state.source='reference';disposePreview();state.modalChart?.dispose();state.modalChart=null;const top=$('.paper-list')?.scrollTop||0;root.innerHTML=`<div class="paper-page-heading"><div><div class="paper-eyebrow">04 / OFFICIAL</div><div class="ui-title-group"><h1>Official results</h1>${helpIcon("Explore recorded paper figures and tables alongside eligible local evaluation results.","Official results help")}</div></div></div><div class="paper-grid ${state.selected[0]==='f'?'figure-selected':''}">${panel('Contents',`<div class="paper-list"><div class="paper-category">FIGURES</div>${VISIBLE_FIGURES.map(([id,title,label])=>`<button type="button" data-paper="f${id}" class="${state.selected===`f${id}`?'active':''}"><span>${label}</span>${esc(title)}</button>`).join('')}<div class="paper-category">TABLES</div>${TABLES.map((title,i)=>`<button type="button" data-paper="t${i+1}" class="${state.selected===`t${i+1}`?'active':''}"><span>Tab.${i+1}</span>${esc(title)}</button>`).join('')}</div>`)}<div class="paper-content" data-paper-content>${contentMarkup()}</div><aside class="paper-panel paper-meta"><div class="paper-head"><div class="ui-title-group"><h2>Source</h2>${helpIcon("Inspect the data source and record identity for this figure or table.","Figure or table source help")}</div></div><div class="paper-pad" data-paper-meta>${sourceMarkup()}</div></aside></div><dialog class="paper-source-dialog" data-paper-dialog></dialog>`;$('.paper-list').scrollTop=top;$('[data-paper-dialog]').addEventListener('close',onDialogClose);mountContentsDrawer();mountPreview()}
 function updateContent(){if(state.selected[0]==='t'&&isInformationalTable(state.selected.slice(1)))state.source='reference';disposePreview();$('[data-paper-content]').innerHTML=contentMarkup();$('[data-paper-meta]').innerHTML=sourceMarkup();mountPreview()}
 async function loadMeasured(){
  if(state.selected[0]!=='t'||isInformationalTable(state.selected.slice(1))){state.source='reference';return}
  const id=++state.request;state.loading=true;state.error='';updateContent();
  try{const payload=await api('/api/paper/tables');if(id!==state.request)return;if(!payload?.tables)throw new Error('Results unavailable');state.liveTables=payload}
  catch(error){if(id===state.request)state.error=error.message||'Results unavailable'}
  finally{if(id===state.request){state.loading=false;if(state.selected[0]==='t'&&state.source==='measured'){updateContent();await awaitPreview()}}}
 }
 function onDialogClose(){state.modalLoad?.cancel();state.modalLoad=null;state.modalChart?.dispose();state.modalChart=null;state.modalData=null;const frame=$('[data-paper-dialog] iframe');if(frame)frame.src='about:blank';$('[data-paper-dialog]')?.replaceChildren()}
 async function onClick(event){
   if(event.target.closest('[data-ui-help]'))return;
  const selection=event.target.closest('[data-paper]');
  if(selection){setContentsDrawerOpen(false);if(mobilePaperQuery.matches)root.scrollTop=0;state.selected=selection.dataset.paper;state.request++;state.loading=false;state.error='';if(state.selected[0]==='t'&&isInformationalTable(state.selected.slice(1)))state.source='reference';await withLoading(async()=>{render();if(state.selected[0]==='t'&&state.source==='measured')await loadMeasured();else await awaitPreview()},{delay:state.selected[0]==='f'||state.selected==='t5'?0:120});requestAnimationFrame(()=>{if(mobilePaperQuery.matches)contentsDrawer?.toggle.focus({preventScroll:true});else $('[data-paper].active')?.scrollIntoView({block:'nearest'})});return}
  const source=event.target.closest('[data-paper-source]');
  if(source){if(source.disabled||isInformationalTable(state.selected.slice(1))&&source.dataset.paperSource==='measured')return;state.source=source.dataset.paperSource;state.request++;state.loading=false;state.error='';await withLoading(async()=>{updateContent();if(state.source==='measured')await loadMeasured();else await awaitPreview()},{delay:state.selected==='t5'?0:120});return}
  const action=event.target.closest('[data-paper-action]')?.dataset.paperAction;
  if(action==='contents-open'){setContentsDrawerOpen(true);return}
  if(action==='contents-close'){setContentsDrawerOpen(false);return}
  if(action==='open-joint-chart'&&state.selected==='t5'&&!state.loading){
   await withLoading(async()=>{const dialog=$('[data-paper-dialog]');state.modalData=tableData();dialog.className='paper-subplot-dialog';dialog.innerHTML=`<div class="paper-dialog-head"><h2>Per-joint MLE · ${state.source==='reference'?'Official':'My results'}</h2>${button('×','close','paper-close','aria-label="Close chart"')}</div><div class="paper-dialog-body"><div class="paper-joint-modal" data-paper-joint-modal></div></div><div class="paper-dialog-foot">${button('Reset','reset')}${button('Close','close')}</div>`;dialog.showModal();state.modalChart=mountJointComparison($('[data-paper-joint-modal]'),state.modalData,{interactive:true});await loadECharts()},{delay:0});
  }else if(action==='close')$('[data-paper-dialog]').close();
  else if(action==='reset'){
   if(state.modalChart){await withLoading(async()=>{state.modalChart.dispose();state.modalChart=mountJointComparison($('[data-paper-joint-modal]'),state.modalData,{interactive:true});await loadECharts()},{delay:0})}
   else{const frame=$('[data-paper-dialog] iframe');if(frame){state.modalLoad?.cancel();await withLoading(()=>{state.modalLoad=assetLoad(frame,frame.src);return state.modalLoad.promise},{delay:0}).catch(error=>{const host=$('.paper-dialog-body');if(host)host.innerHTML=`<div class="paper-empty" role="alert">${esc(error.message)}</div>`})}}
  }
 }
 function onKeyDown(event){const target=event.target.closest('[data-paper-action="open-joint-chart"]');if(!target||!['Enter',' '].includes(event.key))return;event.preventDefault();withLoading(()=>onClick({target})).catch(()=>{})}
 async function onMessage(event){
  const preview=$('[data-paper-content] iframe'),dialog=$('[data-paper-dialog]');if(!state.active||!preview||event.source!==preview.contentWindow||dialog.open)return;
  const data=event.data,id=Number(state.selected.slice(1)),sourceFigure=Number(data?.sourceFigure)||id,item=state.catalog?.[sourceFigure];
  if(!data||data.type!=='paper:open-subplot'||state.selected[0]!=='f'||data.figure!==id||data.renderer!==item?.renderer)return;
  const selected=item.panels?.find(p=>p.id===data.panel&&p.kind!=='image');if(!selected)return;
  const title=[`Figure ${id===9?'9–10':id}`,selected.dataset,selected.sample,selected.label].filter(Boolean).join(' · '),src=data.renderer==='echarts'?`/paper-assets/figure-05/index.html?mode=interactive&panel=${encodeURIComponent(selected.id)}`:`/paper-assets/paper-browser/index.html?figure=${sourceFigure}&mode=interactive&panel=${encodeURIComponent(selected.id)}`;
  dialog.className='paper-subplot-dialog';dialog.innerHTML=`<div class="paper-dialog-head"><h2>${esc(title)}</h2>${button('×','close','paper-close','aria-label="Close subplot"')}</div><div class="paper-dialog-body"><iframe class="paper-modal-frame" title="${esc(title)}"></iframe></div><div class="paper-dialog-foot">${button('Reset','reset')}${button('Close','close')}</div>`;dialog.showModal();
  await withLoading(()=>{state.modalLoad=assetLoad($('[data-paper-dialog] iframe'),src);return state.modalLoad.promise},{delay:0}).catch(error=>{const host=$('.paper-dialog-body');if(dialog.open&&host)host.innerHTML=`<div class="paper-empty" role="alert">${esc(error.message)}</div>`});
 }
 return {setActive(active){const entering=Boolean(active)&&!state.active;state.active=Boolean(active);if(!state.active)setContentsDrawerOpen(false);else if(entering&&mobilePaperQuery.matches)scheduleContentsDrawerOpen();if(active)resizePaperPreview();state.previewChart?.setActive(state.active);state.modalChart?.setActive(state.active);if(active&&state.selected[0]==='t'&&state.source==='measured'&&!isInformationalTable(state.selected.slice(1)))withLoading(loadMeasured).catch(()=>{})},destroy(){state.request++;disposeContentsDrawer();mobilePaperQuery.removeEventListener('change',syncContentsDrawer);disposePreview();state.modalLoad?.cancel();state.modalChart?.dispose();state.modalChart=null;window.removeEventListener('message',onMessage);unbindLoading();root.removeEventListener('keydown',onKeyDown);const dialog=$('[data-paper-dialog]');if(dialog?.open)dialog.close();root.replaceChildren()}};
}
