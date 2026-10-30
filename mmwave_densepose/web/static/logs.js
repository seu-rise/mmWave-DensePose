import {api as defaultApi} from './api.js';
import {withLoading,bindLoadingActions} from './loading.js';
import {ScalarChart,chooseChartMetrics} from './scalar-chart.js';
import {buildRunParameterGroups,parameterLabel,parameterValue} from './log-parameters.js';
import {chartRunIds,selectedUserRuns} from './log-selection.js';
import {helpIcon} from './ui-help.js';

const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const axes=[['step','Step'],['epoch','Epoch'],['relative_time','Relative time'],['wall_time','Wall time']];
const filterOptions={status:['All statuses','recorded','complete','partial','failed','smoke','incomplete','unknown'],integrity:['All integrity states']};
const hasTemporalData=run=>Boolean(run.tags?.length&&(Number(run.sizes?.['history.json'])>0||Number(run.sizes?.['metrics.csv'])>0||run.has_events));
function css(){if(document.querySelector('link[data-logs-css]'))return;const link=document.createElement('link');link.rel='stylesheet';link.href=new URL('./logs.css',import.meta.url).href;link.dataset.logsCss='';document.head.append(link)}
function selectOptions(key){return filterOptions[key].map((x,i)=>`<option value="${i?esc(x):''}">${esc(x)}</option>`).join('')}
function recordsFrom(data){return Array.isArray(data)?data:Array.isArray(data?.runs)?data.runs:[]}
const openIcon='<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M11 3h6v6M17 3l-8 8M8 4H4v12h12v-4"/></svg>';
function valueOf(run,key,...aliases){for(const values of [run.display_metadata,run,run.config])if(values)for(const name of [key,...aliases])if(Object.prototype.hasOwnProperty.call(values,name))return values[name];return null}
function runTime(run){
 const value=valueOf(run,'time','timestamp','started_at','created_at'),source=value??run.training_run_id??run.display_name??'',match=String(source).match(/(\d{4})(\d{2})(\d{2})[_ -]?(\d{2})(\d{2})(\d{2})/);
 if(match)return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}:${match[6]}`;
 const numeric=typeof value==='number'||typeof value==='string'&&/^\d+(\.\d+)?$/.test(value),date=value===null?null:new Date(numeric?Number(value)*(Number(value)<1e12?1000:1):value);
 if(date&&!Number.isNaN(date.valueOf()))return date.toLocaleString('sv-SE',{hour12:false}).replace(',','');
 return '—';
}
function compactNumber(value){if(value===null||value===undefined||value==='')return '—';const number=Number(value);return Number.isFinite(number)?String(number):String(value)}
function deviceLabel(run){const raw=valueOf(run,'device','accelerator');if(raw===null||raw===undefined||raw==='')return '—';const device=String(raw).toUpperCase(),count=Number(valueOf(run,'num_devices','device_count'));return `${device}${count>1?` ×${count}`:''}`}
function runSummary(run){const points=valueOf(run,'num_points','radar_point_size'),dims=valueOf(run,'input_dim','sample_dim','feature_dim'),window=valueOf(run,'seq_len','t_max');return `${compactNumber(points)} pts · ${compactNumber(dims)}D · T${compactNumber(window)}`}
export function paperTableTags(run){
 if(run?.origin!=='official')return [];
 const valid=values=>[...new Set((Array.isArray(values)?values:[]).filter(value=>typeof value==='number'||typeof value==='string'&&/^\d+$/.test(value.trim())).map(Number).filter(value=>Number.isInteger(value)&&value>=1&&value<=10))].sort((a,b)=>a-b);
 const direct=valid(run.paper_tables),dependencies=valid(run.paper_dependency_tables).filter(table=>!direct.includes(table));
 return [...direct.map(table=>({table,dependency:false,label:`Table ${table}`})),...dependencies.map(table=>({table,dependency:true,label:`Table ${table} · PCC`}))].sort((a,b)=>a.table-b.table);
}
function tableTagMarkup(run){return paperTableTags(run).map(tag=>`<span class="logs-run-tag logs-paper-table-tag ${tag.dependency?'is-dependency':''}" title="${tag.dependency?'PCC dependency for':'Official result in'} Table ${tag.table}">${esc(tag.label)}</span>`).join('')}
function runTags(run){const status=String(run.status||'unknown'),seed=valueOf(run,'seed'),epochs=valueOf(run,'epochs','max_epochs','effective_max_epochs'),batch=valueOf(run,'batch_size'),learning=valueOf(run,'learning_rate');return `${tableTagMarkup(run)}<span class="logs-run-tag logs-run-tag-status is-${esc(status.toLowerCase().replace(/[^a-z-]/g,''))}">${esc(status)}</span><span class="logs-run-tag">Seed ${esc(compactNumber(seed))}</span><span class="logs-run-tag">${esc(deviceLabel(run))}</span><span class="logs-run-tag">${esc(compactNumber(epochs))} epochs</span><span class="logs-run-tag">Batch ${esc(compactNumber(batch))}</span><span class="logs-run-tag">LR ${esc(compactNumber(learning))}</span>`}
function detailRows(values){return `<dl class="logs-details-kv">${Object.entries(values).map(([key,value])=>`<dt>${esc(parameterLabel(key))}</dt><dd><code>${esc(parameterValue(value,key))}</code></dd>`).join('')}</dl>`}
function detailGroup(group){const count=Object.keys(group.all).length;return count?`<section class="logs-details-group"><header class="logs-details-group-head"><h3>${esc(group.title)}</h3><button type="button" data-logs-action="parameters-all" data-parameter-group="${esc(group.id)}" aria-label="Show all ${esc(group.title)}">Show all</button></header>${Object.keys(group.primary).length?detailRows(group.primary):`<div class="logs-details-count">${count} parameters</div>`}</section>`:''}

export function mountLogsPanel(root,{api=defaultApi,onTabChange=()=>{},initialOrigin='user'}={}){
 const charts=new Map();let parameterGroups=[];let parameterTrigger=null;
 css();const state={active:true,runs:[],models:[],selected:[],tagsByRun:{},tagOptions:[],selectedTags:[],chartMetrics:[null,null,null,null],series:[],detail:null,details:{},filters:{origin:initialOrigin==='official'?'official':'user'},cursor:null,loading:false,error:'',timer:null,request:0,selectionRequest:0,initialized:false,autoSelect:true,runRequest:0,detailRequest:0,detailTrigger:null};
 root.innerHTML=`<section class="logs-panel" aria-label="Training logs">
 <div class="logs-heading"><div><div class="logs-kicker">02 / LOGS</div><div class="ui-title-group"><h1>Logs</h1>${helpIcon("Browse your training logs or the recorded official training runs.","Training logs help")}</div></div><div class="logs-heading-actions"><button type="button" data-logs-action="tensorboard">Native TensorBoard</button>${helpIcon("Open all indexed training event logs in native TensorBoard.","Native TensorBoard help")}</div></div>
 <nav class="logs-subnav" aria-label="Training panels"><button type="button" data-logs-tab="execution">Execution</button><button type="button" class="active" data-logs-tab="logs" aria-current="page">Logs</button></nav>
 <button class="logs-runs-trigger" type="button" data-logs-action="runs-open" aria-label="Select training runs" aria-controls="logs-runs-drawer" aria-expanded="false"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg></button><dialog class="logs-runs-drawer" id="logs-runs-drawer" data-logs-runs-drawer aria-label="Training runs"><div class="logs-runs-drawer-body" data-logs-runs-drawer-body></div></dialog>
 <div class="logs-layout"><aside class="logs-run-rail"><div class="logs-section-head"><div class="ui-title-group"><h2>Runs</h2>${helpIcon("Select training runs to compare their parameters and scalar curves.","Training runs help")}</div><span class="logs-count" data-logs-count>0 runs</span><button class="logs-runs-close" type="button" data-logs-action="runs-close" aria-label="Close training runs">×</button></div>
 <div class="logs-filterbar"><div class="logs-origin-tabs" role="tablist" aria-label="Origin"><button type="button" role="tab" data-log-origin="user" aria-selected="true">My training</button><button type="button" role="tab" data-log-origin="official" aria-selected="false" tabindex="-1">Official training</button></div><label class="logs-model-filter">Model<select data-log-filter="model" aria-label="Model"><option value="">All models</option></select></label><input type="search" data-log-filter="search" placeholder="Search runs" aria-label="Search runs" autocomplete="off"><details class="logs-extra-filters"><summary>Filters</summary><div class="logs-filter-grid">${['status','integrity'].map(key=>`<label>${key[0].toUpperCase()+key.slice(1)}<select data-log-filter="${key}">${selectOptions(key)}</select></label>`).join('')}</div></details><div class="logs-rail-actions"><span data-selected-count>0 selected</span><button type="button" data-logs-action="select-mine">Select my runs</button><button type="button" class="logs-delete-button" data-logs-action="delete" hidden>Delete selected</button><button type="button" data-logs-action="clear-filters">Clear</button><button type="button" data-logs-action="refresh" aria-label="Refresh runs">↻</button></div></div>
 <div class="logs-run-list" data-run-list><div class="logs-loading">Loading runs…</div></div><button class="logs-more" type="button" data-logs-action="more" hidden>Load more</button></aside>
 <section class="logs-workspace"><div class="logs-workspace-top"><div class="ui-title-group"><span>Scalars</span>${helpIcon("Compare selected runs; use the wheel to zoom and hover to inspect curve values.","Scalar charts help")}</div><div class="logs-controls"><label>X axis <select data-axis>${axes.map(([value,label])=>`<option value="${value}" ${value==='epoch'?'selected':''}>${label}</option>`).join('')}</select></label><label>Smoothing <input type="range" min="0" max="95" step="5" value="60" data-smoothing aria-label="Smoothing"></label><output data-smoothing-value>0.60</output></div></div>
 <div class="logs-chart-grid" data-charts><div class="logs-empty">Select training runs</div></div></section></div>
 <div class="logs-error" data-logs-error role="alert" hidden></div>
 <dialog class="logs-run-dialog" aria-labelledby="logs-run-title"><div class="logs-dialog-head"><div><h2 id="logs-run-title">Training run</h2><div class="logs-details-subtitle" data-run-subtitle></div></div><button type="button" data-logs-action="run-close" aria-label="Close run details">×</button></div><div class="logs-run-dialog-body" data-run-dialog-body></div><div class="logs-tb-actions"><button type="button" data-logs-action="run-close">Close</button></div></dialog></section>`;
 const $=s=>root.querySelector(s);const $$=s=>[...root.querySelectorAll(s)];
 const runsMedia=window.matchMedia('(max-width:850px)'),runsDrawer=$('[data-logs-runs-drawer]'),runsDrawerBody=$('[data-logs-runs-drawer-body]');let drawerResize=0,drawerOpenTimer=0,drawerActivated=false;
 state.runsDrawerOpen=false;
 function runsTrigger(){return $('[data-logs-action="runs-open"]')}
 function resizeCharts(){cancelAnimationFrame(drawerResize);drawerResize=requestAnimationFrame(()=>{drawerResize=0;for(const chart of charts.values())chart.resize()})}
 function syncRunsDrawer(){
  const layout=$('.logs-layout'),rail=layout.querySelector('.logs-run-rail')||runsDrawerBody.querySelector('.logs-run-rail');if(!rail)return;
  if(runsMedia.matches){if(rail.parentElement!==runsDrawerBody)runsDrawerBody.replaceChildren(rail);if(state.runsDrawerOpen&&!runsDrawer.open)runsDrawer.showModal()}
  else{if(rail.parentElement!==layout)layout.prepend(rail);closeRunsDrawer(false)}
  runsTrigger()?.setAttribute('aria-expanded',String(runsDrawer.open));resizeCharts();
 }
 function openRunsDrawer(){clearTimeout(drawerOpenTimer);drawerOpenTimer=0;if(!runsMedia.matches)return;state.runsDrawerOpen=true;syncRunsDrawer();runsTrigger()?.setAttribute('aria-expanded','true')}
 function closeRunsDrawer(restoreFocus=true){clearTimeout(drawerOpenTimer);drawerOpenTimer=0;state.runsDrawerOpen=false;if(runsDrawer.open)runsDrawer.close();runsTrigger()?.setAttribute('aria-expanded','false');if(restoreFocus&&state.active&&runsMedia.matches)runsTrigger()?.focus({preventScroll:true});resizeCharts()}
 runsDrawer.addEventListener('close',()=>{state.runsDrawerOpen=false;runsTrigger()?.setAttribute('aria-expanded','false');resizeCharts()});
 runsDrawer.addEventListener('click',event=>{if(event.target===runsDrawer)closeRunsDrawer()});
 runsMedia.addEventListener('change',syncRunsDrawer);window.addEventListener('resize',resizeCharts);syncRunsDrawer();
 const parameterDialog=document.createElement('dialog');parameterDialog.className='logs-parameters-dialog';parameterDialog.setAttribute('aria-labelledby','logs-parameters-title');
 parameterDialog.innerHTML='<div class="logs-dialog-head"><h2 id="logs-parameters-title">All parameters</h2><button type="button" data-logs-action="parameters-close" aria-label="Close all parameters">×</button></div><div class="logs-parameters-body" data-parameters-body></div><div class="logs-tb-actions"><button type="button" data-logs-action="parameters-close">Close</button></div>';
 $('.logs-panel').appendChild(parameterDialog);
 const deleteDialog=document.createElement('dialog');deleteDialog.className='logs-delete-dialog';deleteDialog.setAttribute('aria-labelledby','logs-delete-title');
 deleteDialog.innerHTML='<div class="logs-dialog-head"><h2 id="logs-delete-title">Delete training logs</h2><button type="button" data-logs-action="delete-cancel" aria-label="Cancel deletion">×</button></div><div class="logs-delete-body"><p data-delete-summary></p><ul data-delete-runs></ul><p class="logs-delete-note">Model checkpoints and test results are kept. Logs are moved to local trash for recovery.</p><p data-delete-error role="alert" hidden></p></div><div class="logs-tb-actions"><button type="button" data-logs-action="delete-cancel">Cancel</button><button type="button" class="logs-delete-button" data-logs-action="delete-confirm">Delete logs</button></div>';
 $('.logs-panel').appendChild(deleteDialog);
 let deleteIds=[],deleting=false;
 deleteDialog.addEventListener('cancel',event=>{if(deleting)event.preventDefault()});
 function selectedForDeletion(){return selectedUserRuns(state.runs,state.selected,state.details)}
 function openDeleteDialog(){
  const runs=selectedForDeletion();if(!runs.length||deleting)return;
  deleteIds=runs.map(run=>run.run_id||run.id);
  $('[data-delete-summary]').textContent=`Delete ${runs.length} selected training log${runs.length===1?'':'s'}?`;
  $('[data-delete-runs]').innerHTML=runs.map(run=>`<li><strong>${esc(run.model||'Training run')}</strong><span>${esc(run.training_run_id||run.display_name||run.id)}</span></li>`).join('');
  $('[data-delete-error]').hidden=true;deleteDialog.showModal();
 }
 async function deleteSelected(){
  if(deleting||!deleteIds.length)return;deleting=true;
  deleteDialog.querySelectorAll('button').forEach(button=>button.disabled=true);
  try{
   const result=await api('/api/logs/runs/delete',{method:'POST',body:{run_ids:deleteIds}}),removed=new Set(result.deleted_run_ids);
   state.request++;state.selectionRequest++;state.detailRequest++;state.runRequest++;
   state.autoSelect=false;state.selected=state.selected.filter(id=>!removed.has(id));
   state.runs=state.runs.filter(run=>!removed.has(run.run_id||run.id));
   for(const id of removed){delete state.details[id];delete state.tagsByRun[id]}
   state.series=state.series.filter(item=>!removed.has(item.run_id));state.detail=null;
   deleteDialog.close();if($('.logs-run-dialog').open)$('.logs-run-dialog').close();
   renderRuns();renderTags();renderCharts();await loadRuns();
  }catch(error){const message=$('[data-delete-error]');message.textContent=error.message||'Unable to delete logs';message.hidden=false}
  finally{deleting=false;deleteDialog.querySelectorAll('button').forEach(button=>button.disabled=false);renderRuns()}
 }
 parameterDialog.addEventListener('close',()=>{$('[data-parameters-body]').replaceChildren();if(parameterTrigger?.isConnected)parameterTrigger.focus();parameterTrigger=null});
 function openAllParameters(id,trigger){const group=parameterGroups.find(item=>item.id===id);if(!group)return;parameterTrigger=trigger;$('#logs-parameters-title').textContent=group.title+' — All parameters';$('[data-parameters-body]').innerHTML=detailRows(group.all);if(!parameterDialog.open)parameterDialog.showModal()}
 function showError(message=''){state.error=message;const el=$('[data-logs-error]');el.hidden=!message;el.textContent=message}
 function query(){const params=new URLSearchParams({limit:'500',task:'train'});for(const [k,v] of Object.entries(state.filters))if(v)params.set(k,v);if(state.cursor)params.set('cursor',state.cursor);return params.toString()}
 function renderModels(){const select=$('[data-log-filter="model"]'),selected=state.filters.model||'',models=[...new Set([...state.models,...(selected?[selected]:[])])].sort();select.innerHTML='<option value="">All models</option>'+models.map(model=>`<option value="${esc(model)}" ${model===selected?'selected':''}>${esc(model)}</option>`).join('');select.value=selected}
 function renderRuns(){
  for(const tab of $$('[data-log-origin]')){const active=tab.dataset.logOrigin===state.filters.origin;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1}
  renderModels();const list=$('[data-run-list]');$('[data-logs-count]').textContent=`${Number(state.total??state.runs.length).toLocaleString()} runs`;$('[data-selected-count]').textContent=`${state.selected.length} selected${state.selected.length>8?' · 8 plotted':''}`;const deletable=selectedForDeletion(),deleteButton=$('[data-logs-action="delete"]');deleteButton.hidden=!deletable.length;deleteButton.disabled=deleting;deleteButton.textContent=`Delete selected (${deletable.length})`;$('[data-logs-action="select-mine"]').disabled=deleting||!state.runs.some(run=>run.origin==='user'&&run.can_delete===true);$('[data-logs-action="more"]').hidden=!state.nextCursor;
  const integrity=$('[data-log-filter="integrity"]'),selectedIntegrity=state.filters.integrity||'',integrityValues=new Set(state.runs.map(run=>run.integrity).filter(Boolean));if(selectedIntegrity)integrityValues.add(selectedIntegrity);integrity.innerHTML=`<option value="">All integrity states</option>${[...integrityValues].sort().map(value=>`<option value="${esc(value)}" ${value===selectedIntegrity?'selected':''}>${esc(value)}</option>`).join('')}`;
  if(!state.runs.length){list.innerHTML='<div class="logs-empty-rail">No runs found</div>';return}
  list.innerHTML=state.runs.map(run=>{const id=run.run_id||run.id,index=state.selected.indexOf(id),checked=index>=0,time=runTime(run),original=run.training_run_id||run.display_name||id,origin=run.origin==='official'?'Official':'My training';return `<article class="logs-run-row ${checked?'is-selected':''}"><label class="logs-run-selector"><input type="checkbox" data-run-select value="${esc(id)}" aria-label="Select ${esc(run.model||'training run')} ${esc(time)} ${esc(original)}" ${checked?'checked':''}><span class="logs-run-copy"><span class="logs-run-topline"><strong class="${checked&&index<8?'logs-run-color-'+index:''}">${esc(run.model||'Training run')}</strong><span class="logs-run-origin">${esc(origin)}</span></span><time class="logs-run-time">${esc(time)}</time><span class="logs-run-name" title="${esc(original)}">${esc(runSummary(run))}</span><span class="logs-run-tags">${runTags(run)}</span></span></label><button type="button" class="logs-run-open" data-run-open="${esc(id)}" aria-label="Open run details ${esc(original)}" title="Open run details">${openIcon}</button></article>`}).join('');
 }
 async function loadRuns({append=false}={}){if(!state.active)return;const request=++state.runRequest;state.loading=true;showError();if(!append){state.cursor=null;state.nextCursor=null;state.runs=[];$('[data-run-list]').innerHTML='<div class="logs-loading">Loading runs…</div>'}try{const data=await api(`/api/logs/runs?${query()}`);if(request!==state.runRequest)return;state.models=(data.models||[]).map(model=>typeof model==='string'?model:model.id).filter(Boolean);const rows=recordsFrom(data).filter(run=>(run.action||run.task)==='train');state.runs=append?[...state.runs,...rows]:rows;state.total=data?.total;state.nextCursor=data?.next_cursor??data?.cursor??null;state.initialized=true;if(state.autoSelect&&!state.selected.length){state.selected=state.runs.filter(hasTemporalData).slice(0,2).map(run=>run.run_id||run.id);if(state.selected.length)state.autoSelect=false}renderRuns();await loadTagsAndDetail()}catch(error){if(request!==state.runRequest)return;showError(error.message||String(error));if(!state.runs.length)$('[data-run-list]').innerHTML='<div class="logs-empty-rail">Unable to load runs</div>'}finally{if(request===state.runRequest)state.loading=false}}
 function renderTags(){
  state.tagOptions=[...new Set(Object.values(state.tagsByRun).flatMap(tags=>tags||[]))].sort((a,b)=>String(a).localeCompare(String(b)));
  if(state.tagOptions.length)state.chartMetrics=chooseChartMetrics(state.tagOptions,state.chartMetrics);
  state.selectedTags=[...new Set(state.chartMetrics.filter(tag=>state.tagOptions.includes(tag)))];
 }
 function renderCharts(){
  const area=$('[data-charts]'),message=!state.selected.length?'Select training runs':!state.tagOptions.length?'No scalar tags':null;
  if(message){for(const chart of charts.values())chart.destroy();charts.clear();area.innerHTML=`<div class="logs-empty">${message}</div>`;return}
  for(const child of [...area.children])if(!child.classList.contains('logs-chart-card'))child.remove();
  area.style.setProperty('--logs-chart-rows','2');
  state.chartMetrics.forEach((tag,index)=>{
   let chart=charts.get(index);
   if(!chart){chart=new ScalarChart({tag:tag||'',chartIndex:index,onMetricChange:value=>withLoading(()=>selectChartMetric(index,value))});charts.set(index,chart)}
   if(area.children[index]!==chart.element)area.insertBefore(chart.element,area.children[index]||null);
   chart.setMetric(tag,state.tagOptions);chart.setActive(state.active);chart.update(state.series.filter(item=>item.tag===tag),chartRunIds(state.selected));
  });
 }
 async function selectChartMetric(index,tag){
  if(tag&&!state.tagOptions.includes(tag))return;
  state.chartMetrics[index]=tag||null;
  state.selectedTags=[...new Set(state.chartMetrics.filter(Boolean))];
  renderCharts();await loadSeries();
 }
 function clearRunSelection(){state.chartMetrics=[null,null,null,null];state.selectedTags=[];state.selected=[];state.autoSelect=true;state.details={};state.tagsByRun={};state.series=[];state.request++;state.selectionRequest++;state.runs=[];state.total=0;renderRuns();renderTags();renderCharts()}
 function renderRunDetails(run){
  parameterGroups=buildRunParameterGroups({...run,time:runTime(run)});
  $('[data-run-subtitle]').textContent=[run.model,runTime(run)].filter(value=>value&&value!=='—').join(' · ');
  const tableTags=tableTagMarkup(run);
  $('[data-run-dialog-body]').innerHTML=`${tableTags?`<div class="logs-details-paper-tags" aria-label="Official tables">${tableTags}</div>`:''}<div class="logs-details-parameter-grid">${parameterGroups.map(detailGroup).join('')}</div>`;
 }
 async function openRunDetails(id,trigger){
  const request=++state.detailRequest,dialog=$('.logs-run-dialog');state.detailTrigger=trigger;$('[data-run-subtitle]').textContent='';$('[data-run-dialog-body]').innerHTML='<div class="logs-loading">Loading</div>';if(!dialog.open)dialog.showModal();
  try{const run=await api(`/api/logs/run?run_id=${encodeURIComponent(id)}`);if(request!==state.detailRequest||!dialog.open)return;renderRunDetails(run)}catch(error){if(request===state.detailRequest&&dialog.open)$('[data-run-dialog-body]').innerHTML=`<div class="logs-error" role="alert">${esc(error.message||'Run details unavailable')}</div>`}
 }
 $('.logs-run-dialog').addEventListener('close',()=>{if(parameterDialog.open)parameterDialog.close();parameterGroups=[];state.detailRequest++;$('[data-run-dialog-body]').replaceChildren();if(state.detailTrigger?.isConnected)state.detailTrigger.focus();state.detailTrigger=null});
 async function loadTagsAndDetail(){if(!state.active)return;const request=++state.selectionRequest;state.tagsByRun={};state.series=[];state.detail=null;state.details={};renderTags();renderCharts();if(!state.selected.length)return;try{const ids=chartRunIds(state.selected),results=await Promise.all([api(`/api/logs/tags?run_id=${ids.map(encodeURIComponent).join(',')}`),...ids.map(id=>api(`/api/logs/run?run_id=${encodeURIComponent(id)}`))]);if(request!==state.selectionRequest)return;for(const run of recordsFrom(results[0]))state.tagsByRun[run.run_id]=run.tags||[];state.details=Object.fromEntries(ids.map((id,index)=>[id,results[index+1]]));state.detail=results[1];renderTags();await loadSeries()}catch(error){if(request===state.selectionRequest)showError(error.message||String(error))}}
 async function loadSeries(){if(!state.active||!state.selected.length||!state.selectedTags.length){state.series=[];renderCharts();return}const generation=++state.request;try{const data=await api('/api/logs/series',{method:'POST',body:{run_ids:chartRunIds(state.selected),tags:state.selectedTags,x_axis:$('[data-axis]').value,smoothing:Number($('[data-smoothing]').value)/100,max_points:1200}});if(generation!==state.request)return;state.series=(data?.series||[]).map(item=>{const run=state.details[item.run_id]||state.runs.find(row=>(row.run_id||row.id)===item.run_id);const time=run?runTime(run):'—',seed=run?valueOf(run,'seed'):null,shortTime=/^\d{4}-/.test(time)?time.slice(5):time;return {...item,name:run?[run.model,run.training_run_id].filter(Boolean).join(' · '):item.name,legend_name:run?[run.model,shortTime==='—'?null:shortTime,seed===null||seed===undefined?null:`s${seed}`].filter(Boolean).join(' · '):item.name}});renderCharts()}catch(error){showError(error.message||String(error));state.series=[];renderCharts()}}
 function openTensorboard(){
  const url=new URL('./tensorboard.html',import.meta.url);
  const token=new URLSearchParams(location.hash.slice(1)).get('token');
  if(token)url.hash=new URLSearchParams({token}).toString();
  window.open(url.href,'_blank','noopener,noreferrer');
 }
 async function selectOrigin(origin){
  if(!['user','official'].includes(origin)||origin===state.filters.origin)return;
  state.filters.origin=origin;state.filters.model='';clearRunSelection();await loadRuns();
 }
 const onOriginKey=event=>{
  const tab=event.target.closest('[data-log-origin]');if(!tab)return;
  if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
   event.preventDefault();const origin=event.key==='Home'?'user':event.key==='End'?'official':tab.dataset.logOrigin==='user'?'official':'user';
   $(`[data-log-origin="${origin}"]`).focus();withLoading(()=>selectOrigin(origin));
  }
 };
 root.addEventListener('keydown',onOriginKey);
 const removeLoadingClick=bindLoadingActions(root,{click:async e=>{if(e.target.closest('[data-ui-help]'))return;const origin=e.target.closest('[data-log-origin]');if(origin){await selectOrigin(origin.dataset.logOrigin);return}const run=e.target.closest('[data-run-open]');if(run){e.preventDefault();e.stopPropagation();await openRunDetails(run.dataset.runOpen,run);return}const tab=e.target.closest('[data-logs-tab]');if(tab){if(tab.dataset.logsTab==='execution')onTabChange('execution');return}const actionButton=e.target.closest('[data-logs-action]'),action=actionButton?.dataset.logsAction;if(!action)return;if(action==='runs-open'){openRunsDrawer();return}if(action==='runs-close'){closeRunsDrawer();return}if(action==='delete')openDeleteDialog();else if(action==='delete-cancel'){if(!deleting)deleteDialog.close()}else if(action==='delete-confirm')await deleteSelected();else if(action==='select-mine'){state.autoSelect=false;state.selected=[...new Set([...state.selected,...selectedUserRuns(state.runs,state.runs.map(run=>run.run_id||run.id)).map(run=>run.run_id||run.id)])].slice(0,500);renderRuns();await loadTagsAndDetail()}else if(action==='parameters-all')openAllParameters(actionButton.dataset.parameterGroup,actionButton);else if(action==='parameters-close')parameterDialog.close();else if(action==='run-close')$('.logs-run-dialog').close();else if(action==='refresh')await loadRuns();else if(action==='more'&&state.nextCursor){state.cursor=state.nextCursor;await loadRuns({append:true})}else if(action==='clear-filters'){clearRunSelection();state.filters={origin:state.filters.origin};$$('[data-log-filter]').forEach(el=>el.value='');await loadRuns()}else if(action==='tensorboard')openTensorboard()}});
 const removeLoadingChange=bindLoadingActions(root,{change:async e=>{const el=e.target;if(el.matches('[data-log-filter]')&&el.type!=='search'){const key=el.dataset.logFilter,value=el.value;state.filters[key]=value;if(['origin','model'].includes(key))clearRunSelection();await loadRuns()}else if(el.matches('[data-run-select]')){state.autoSelect=false;const id=el.value;if(el.checked){if(state.selected.length>=500){el.checked=false;showError('Select up to 500 runs at once');return}state.selected.push(id)}else state.selected=state.selected.filter(x=>x!==id);renderRuns();await loadTagsAndDetail()}else if(el.matches('[data-axis]'))await loadSeries();}});
 root.addEventListener('input',e=>{if(e.target.matches('[data-log-filter="search"]')){clearTimeout(state.searchTimer);state.searchTimer=setTimeout(()=>{state.filters.search=e.target.value;loadRuns()},220)}else if(e.target.matches('[data-smoothing]')){$('[data-smoothing-value]').value=(Number(e.target.value)/100).toFixed(2);loadSeries()}});
 async function poll(){clearTimeout(state.timer);if(!state.active)return;await loadSeries();if(state.active)state.timer=setTimeout(poll,5000)}
 async function activate(active=true){const entering=Boolean(active)&&(!state.active||!drawerActivated);state.active=Boolean(active);if(!state.active)closeRunsDrawer(false);else{drawerActivated=true;if(entering&&runsMedia.matches){clearTimeout(drawerOpenTimer);drawerOpenTimer=setTimeout(()=>{drawerOpenTimer=0;if(state.active&&runsMedia.matches)openRunsDrawer()},350)}}for(const chart of charts.values())chart.setActive(state.active);clearTimeout(state.timer);if(state.active){if(!state.initialized)await loadRuns();else if(!state.tagOptions.length&&state.selected.length)await loadTagsAndDetail();state.timer=setTimeout(poll,5000)}else{state.request++;}}
 state.total=0;state.nextCursor=null;loadRuns();return {activate,setActive:activate,destroy(){closeRunsDrawer(false);runsMedia.removeEventListener('change',syncRunsDrawer);window.removeEventListener('resize',resizeCharts);cancelAnimationFrame(drawerResize);root.removeEventListener('keydown',onOriginKey);removeLoadingClick();removeLoadingChange();for(const chart of charts.values())chart.destroy();charts.clear();state.active=false;state.request++;state.selectionRequest++;clearTimeout(state.timer);clearTimeout(state.searchTimer);state.runRequest++;state.detailRequest++;parameterDialog.close();deleteDialog.close();$('.logs-run-dialog').close();root.replaceChildren()}};
}
