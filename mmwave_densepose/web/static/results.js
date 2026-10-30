import * as THREE from "./vendor/three.module.js";
import {GLTFLoader} from "./vendor/loaders/GLTFLoader.js";
import {SpatialView,PAGES_RENDER_CONFIG,createGlowTexture} from "./scene.js";
import {mountJointDiagram} from "./joint-diagram.js";
import {paperTableTags} from "./logs.js";
import {withLoading,bindLoadingActions} from "./loading.js";
import {createPaperAssetDownload,officialAssetId,paperAssetStatusHtml} from "./paper-assets.js";
import {createHelpIcon} from "./ui-help.js";
import {FramePrefetch} from "./frame-prefetch.js";

const PLAYBACK_INTERVAL=1000/6.5;

const BONES=[[0,1],[1,18],[18,2],[2,3],[18,4],[4,5],[5,6],[18,7],[7,8],[8,9],[0,10],[10,11],[11,12],[12,13],[0,14],[14,15],[15,16],[16,17]];
const JOINTS=["SpineBase","SpineMid","Neck","Head","ShoulderLeft","ElbowLeft","HandLeft","ShoulderRight","ElbowRight","HandRight","HipLeft","KneeLeft","AnkleLeft","FootLeft","HipRight","KneeRight","AnkleRight","FootRight","SpineShoulder"];
const DISPLAY_JOINTS=["Spine base","Spine mid","Neck","Head","Shoulder L","Elbow L","Hand L","Shoulder R","Elbow R","Hand R","Hip L","Knee L","Ankle L","Foot L","Hip R","Knee R","Ankle R","Foot R","Spine shoulder"];
const compareColors=["#4ec3ff","#eabc78","#c19fef","#78d9b2","#ff9b9b","#91d7e3","#ed8796","#a6da95"];
const create=(tag,cls,text)=>{const x=document.createElement(tag);if(cls)x.className=cls;if(text!==undefined&&text!==null)x.textContent=String(text);return x;};
const actionButton=(label,action,cls)=>{const x=create("button",cls||"",label);x.type="button";x.dataset.action=action;return x;};
const esc=v=>String(v==null?"":v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(x,d=2)=>x===null||x===undefined||!Number.isFinite(Number(x))?"-":Number(x).toFixed(d);
const list=x=>Array.isArray(x)?x:[];
const finite=p=>Array.isArray(p)&&p.length>=3&&p.slice(0,3).every(Number.isFinite);
function ensureStyles(){if(document.getElementById("results-css"))return;const l=document.createElement("link");l.id="results-css";l.rel="stylesheet";l.href=new URL("./results.css",import.meta.url).href;document.head.append(l)}
function metricCard(label,value,unit){const card=create("article","results-metric");card.append(create("span","results-metric-label",label),create("strong","results-metric-value",fmt(value,unit==="%"?1:2)),create("small","results-metric-unit",unit));return card}
function badge(text,cls=""){return create("span","results-badge "+cls,text)}
export function canDeleteResult(record,source){return source==="user"&&record?.source==="user"&&record.can_delete!==false&&Boolean(record.record_id)}

export async function mountResultsPanel(root,{api,onViewChange=()=>{},view=root.dataset.view||"results",initialSource="user"}={}){
  ensureStyles();root.classList.add("results-root");
  const state={source:initialSource==="official"?"official":"user",loading:false,error:"",predictionLoadingId:null,predictionErrorId:null,active:false,diagram:null,models:[],weights:[],records:[],model:null,weight:null,record:null,view,
    detail:"joints",samples:[],sampleIndex:0,sampleTotal:0,sampleOffset:0,sequenceIds:[],sequenceCounts:{},sequence:"",request:0,sampleRequest:0,
    selected:new Set(),compareResult:null,compareRecords:[],deletePending:false,deleteDialog:null,disposed:false,scene:null,prediction:null,avatar:null,avatarModels:[],avatarModelId:null,avatarContext:null,avatarRequest:0,avatarLoadingId:null,avatarLoadingPromise:null,avatarStatus:"",playing:false,timer:null,
    points:true,gt:true,predictionVisible:true,avatarVisible:false,avatarSource:"prediction",avatarOpacity:.78,fitted:false,baseline:null,compareRequest:0,compareError:"",selectionDrawerOpen:false};
  const mobileResultsQuery=window.matchMedia("(max-width:850px)");
  let selectionDrawer=null,selectionDrawerOpenTimer=0;
  let frameGeneration=0,playbackRun=0,frameController=null;
  const samplePages=new Map();
  // Keep avatar conversion in sample order while buffering ahead of playback.
  const framePrefetch=new FramePrefetch(prepareSample,{maxFrames:8,maxBytes:16*1024**2,concurrency:1});
  const predictionDownload=createPaperAssetDownload({api,onChange:renderPredictionDownload,onInstalled:async query=>{if(state.source==='official'&&officialAssetId(state.record)===query.asset_id&&state.record?.test_id===query.test_id)await loadRecord({loadVisual:false})}});
  function predictionDownloadHtml(){
    const status=paperAssetStatusHtml(predictionDownload.state,{title:'Official prediction PKL',downloadLabel:'Download predictions'});
    const pkl=state.record?.prediction_pkl;
    return status+(state.predictionNeedsOpen&&pkl?.present&&(pkl.pending||pkl.valid)?'<button type="button" class="button button-quiet results-prediction-open" data-action="open-prediction">Open visualization</button>':'');
  }
  function renderPredictionDownload(){const host=root.querySelector('[data-official-prediction-download]');if(host)host.innerHTML=predictionDownloadHtml()}
  function syncPredictionDownload(){const id=state.source==='official'&&state.record?.source==='official'?officialAssetId(state.record):null;return predictionDownload.select(id&&state.record?.test_id?{kind:'predictions',asset_id:id,test_id:state.record.test_id}:null)}
  const removeLoadingActions=bindLoadingActions(root,{change:onChange,click:onClick});root.addEventListener("input",onInput);mobileResultsQuery.addEventListener("change",syncSelectionDrawer);
  await loadModels();
  await loadAvatarModels();
  if(state.view==="compare"){await loadCompareRecords();render()}

  function resultsApi(path,options={}){
    if(options.method==="POST")return api(path,{...options,body:{...options.body,source:state.source}});
    return api(path+(path.includes("?")?"&":"?")+new URLSearchParams({source:state.source}),options);
  }
  async function loadModels(){
    const previousModel=state.model,previousRecord=state.record?.record_id;
    const request=++state.request;state.loading=true;state.error="";render();
    try{
      const payload=await resultsApi("/api/results/models");
      if(request!==state.request)return;state.loading=false;state.models=list(payload.models);
      if(!state.models.length){renderEmpty();return}
      if(state.view==="compare"){render();return}
      const preserve=Boolean(previousRecord&&state.models.some(model=>model.id===previousModel));
      state.model=preserve?previousModel:(state.models.find(model=>model.id==="PASTNet"&&model.test_count>0)||state.models.find(model=>model.test_count>0)||state.models.find(model=>model.id==="PASTNet")||state.models[0]).id;
      await loadWeights({preserveSelection:preserve});
    }catch(error){if(request===state.request)renderError(error)}
  }
  async function loadAvatarModels(){
    try{
      const payload=await api("/api/avatar/models");state.avatarModels=list(payload.models).filter(model=>model.id==="xbot");
      state.avatarModelId=state.avatarModels.length?"xbot":null;
      if(state.view!=="compare"&&state.record&&hasPkl())render();
    }catch(_error){state.avatarModels=[];state.avatarModelId=null}
  }
  async function loadWeights({preserveSelection=false}={}){
    const previousWeight=state.weight;
    const request=++state.request;disposeVisual();state.loading=true;render();
    try{
      const payload=await resultsApi("/api/results/models/"+encodeURIComponent(state.model)+"/weights");
      if(request!==state.request)return;
      state.loading=false;state.weights=list(payload.weights);state.weight=(preserveSelection&&state.weights.find(weight=>weight.id===previousWeight)||state.weights.find(weight=>list(weight.tests).some(test=>test.status==="complete"))||state.weights.find(weight=>list(weight.tests).length>0&&list(weight.paper_tables).some(table=>Number(table)===4))||state.weights.find(weight=>list(weight.tests).length>0)||state.weights[0])?.id||null;
      if(!state.weight){state.records=[];state.record=null;render();return}
      await loadTests({preserveSelection});
    }catch(error){if(request===state.request)renderError(error)}
  }
  async function loadTests({preserveSelection=false}={}){
    const previousRecord=preserveSelection?state.record?.record_id:null;
    const weight=state.weights.find(x=>x.id===state.weight);
    if(!weight){state.records=[];state.record=null;render();return}
    const request=++state.request;disposeVisual();state.loading=true;render();
    const training=weight.training_run_id||weight.id;
    try{
      const payload=await resultsApi("/api/results/models/"+encodeURIComponent(state.model)+"/weights/"+encodeURIComponent(training)+"/tests");
      if(request!==state.request)return;
      state.loading=false;state.records=list(payload.tests).filter(x=>x.model===state.model&&(x.weight_id===weight.id||x.training_run_id===training));
      state.record=state.records.find(record=>record.record_id===previousRecord)||state.records.find(record=>record.complete)||state.records[0]||null;state.samples=[];state.sampleIndex=0;state.sampleTotal=0;
      await loadRecord();
    }catch(error){if(request===state.request)renderError(error)}
  }
  async function loadRecord({loadVisual=true}={}){
    disposeVisual();state.lastPayload=null;state.sampleRequest++;state.predictionLoadingId=null;state.predictionErrorId=null;
    predictionDownload.select(null);
    if(!state.record){render();return}
    const request=++state.request,id=state.record.record_id;
    try{
      const payload=await resultsApi("/api/results/record?"+new URLSearchParams({record_id:id}));
      if(request!==state.request)return;
      state.record=payload;state.samples=[];state.sampleIndex=0;state.sampleTotal=0;state.predictionNeedsOpen=!loadVisual;syncPredictionDownload();
      if(!loadVisual){render();return}
      if(hasPkl())await loadSamples();else if(payload.prediction_pkl?.pending)await loadPendingPrediction(id,request);else render();
    }catch(error){if(request===state.request)renderError(error)}
  }
  async function loadPendingPrediction(recordId,request){
    state.predictionLoadingId=recordId;render();
    try{
      await resultsApi("/api/results/samples?"+new URLSearchParams({record_id:recordId,limit:"1"}));
      if(request!==state.request||state.record?.record_id!==recordId)return;
      await loadRecord();
    }catch(_error){
      if(request!==state.request||state.record?.record_id!==recordId)return;
      state.predictionLoadingId=null;state.predictionErrorId=recordId;render();
    }
  }
  function hasPkl(){const p=state.record?.prediction_pkl;return Boolean(p?.present&&p?.available&&p?.valid)}
  function isCompletion(){return state.record?.task==="completion"||state.record?.model==="PCCCompletion"}
  function sampleQuery(offset=0,sequence=state.sequence){const query=new URLSearchParams({record_id:state.record.record_id,limit:"500",offset:String(offset)});if(sequence)query.set("sequence_id",sequence);return query}
  async function loadSamples(preserveSequence=false){
    if(!hasPkl()){render();return}
    stopPlayback();samplePages.clear();
    const request=state.request,sequence=++state.sampleRequest,id=state.record.record_id;
    try{
      let payload=await resultsApi("/api/results/samples?"+sampleQuery(0,""));
      if(request!==state.request||sequence!==state.sampleRequest||state.record?.record_id!==id)return;
      state.sequenceIds=list(payload.sequence_ids);state.sequenceCounts=payload.sequence_counts||{};
      if(!preserveSequence)state.sequence=state.sequenceIds[0]||"";
      if(state.sequence){payload=await resultsApi("/api/results/samples?"+sampleQuery());if(request!==state.request||sequence!==state.sampleRequest||state.record?.record_id!==id)return}
      state.samples=list(payload.samples);state.sampleTotal=Number(payload.total)||0;state.sampleOffset=Number(payload.offset)||0;state.sampleIndex=0;render();
      rememberSamplePage(0,payload);
      if(state.samples.length)await loadSample(0);
    }catch(error){if(request===state.request)renderError(error)}
  }
  function currentSample(){return state.samples[state.sampleIndex-state.sampleOffset]}
  function rememberSamplePage(offset,page){
    samplePages.set(offset,Promise.resolve(page));
    while(samplePages.size>3)samplePages.delete(samplePages.keys().next().value);
  }
  async function prepareSample(index,generation,signal){
    const recordId=state.record?.record_id,offset=Math.floor(index/500)*500;
    let pending=samplePages.get(offset);
    if(!pending&&offset===state.sampleOffset&&state.samples.length){
      pending=Promise.resolve({samples:state.samples,offset});samplePages.set(offset,pending);
    }
    if(!pending){
      pending=resultsApi("/api/results/samples?"+sampleQuery(offset),{signal});
      samplePages.set(offset,pending);
      while(samplePages.size>3)samplePages.delete(samplePages.keys().next().value);
      pending.catch(()=>{if(samplePages.get(offset)===pending)samplePages.delete(offset)});
    }
    const page=await pending;
    if(signal.aborted||generation!==frameGeneration||state.record?.record_id!==recordId)throw new DOMException("Sample request cancelled","AbortError");
    const sampleId=list(page.samples)[index-offset]?.sample_id;
    if(!sampleId)throw new Error("Sample unavailable");
    const payload=await resultsApi("/api/results/sample?"+new URLSearchParams({record_id:recordId,sample_id:sampleId}),{signal});
    if(!payload||!Array.isArray(payload.radar)||!Array.isArray(payload.target)||!Array.isArray(payload.prediction))throw new Error("Sample unavailable");
    return {payload,page,bytes:4096+payload.radar.length*96,release(){}};
  }
  function stopPlayback(){
    state.playing=false;playbackRun++;frameGeneration++;clearTimeout(state.timer);state.timer=null;
    frameController?.abort();frameController=null;framePrefetch.clear();samplePages.clear();updatePlay();
  }
  async function startPlayback(){
    stopPlayback();if(state.sampleTotal<2||!hasPkl()||state.disposed)return;
    state.playing=true;updatePlay();const run=playbackRun;
    const next=(state.sampleIndex+1)%state.sampleTotal;
    framePrefetch.begin(frameGeneration,next,state.sampleTotal-1,64*1024);
    await framePrefetch.warm();
    if(run!==playbackRun||!state.playing)return;
    async function tick(){
      if(run!==playbackRun||!state.playing)return;
      const started=performance.now(),index=(state.sampleIndex+1)%state.sampleTotal;
      if(index===0)framePrefetch.begin(frameGeneration,0,state.sampleTotal-1,64*1024);
      const loaded=await loadSample(index,{playback:true});
      if(run!==playbackRun||!state.playing)return;
      if(!loaded){stopPlayback();return}
      state.timer=setTimeout(tick,Math.max(0,PLAYBACK_INTERVAL-(performance.now()-started)));
    }
    state.timer=setTimeout(tick,PLAYBACK_INTERVAL);
  }
  async function loadSample(index,{playback=false}={}){
    if(index<0||index>=state.sampleTotal||!hasPkl())return;
    if(!playback)stopPlayback();
    const recordId=state.record.record_id,token=++state.sampleRequest,generation=frameGeneration;
    frameController?.abort();const controller=new AbortController();frameController=controller;
    if(!playback)setCanvasStatus("Loading");
    try{
      const frame=await (playback?framePrefetch.take(index,generation,controller.signal):prepareSample(index,generation,controller.signal));
      if(controller.signal.aborted||generation!==frameGeneration||token!==state.sampleRequest||state.record?.record_id!==recordId){frame.release();return false}
      state.samples=list(frame.page.samples);state.sampleOffset=Number(frame.page.offset)||0;
      state.sampleIndex=index;drawFrame(frame.payload);frame.release();return true;
    }catch(error){if(error.name!=="AbortError"&&token===state.sampleRequest&&generation===frameGeneration)setCanvasStatus("Sample unavailable");return false}
    finally{if(frameController===controller)frameController=null}
  }
  async function seekTargetFrame(target){
    if(!Number.isInteger(target)||target<0||!hasPkl())return;
    stopPlayback();
    const chosen=state.sequence||currentSample()?.sequence_id;if(!chosen)return;
    const recordId=state.record.record_id,token=++state.sampleRequest;setCanvasStatus("Loading");
    try{
      const total=Number(state.sequenceCounts[chosen])||state.sampleTotal;let chosenPage=null,local=-1;
      for(let offset=0;offset<total;offset+=500){const page=await resultsApi("/api/results/samples?"+sampleQuery(offset,chosen));if(token!==state.sampleRequest||state.record?.record_id!==recordId)return;local=list(page.samples).findIndex(item=>item.target_frame===target);if(local>=0){chosenPage=page;break}const last=list(page.samples).at(-1);if(last?.target_frame>target)break}
      if(!chosenPage){setCanvasStatus("Target frame unavailable");return}
      state.sequence=chosen;state.samples=list(chosenPage.samples);state.sampleOffset=Number(chosenPage.offset)||0;state.sampleTotal=Number(chosenPage.total)||total;const slider=root.querySelector("[data-scrubber]");if(slider)slider.max=Math.max(0,state.sampleTotal-1);await loadSample(state.sampleOffset+local);
    }catch(_error){if(token===state.sampleRequest)setCanvasStatus("Target frame unavailable")}
  }
  function disposeVisual(){
    state.diagram?.dispose();state.diagram=null;state.sampleRequest++;stopPlayback();
    state.avatarRequest++;state.avatarLoadingId=null;state.avatarLoadingPromise=null;disposeAvatarContext();
    if(state.scene){state.scene.dispose();state.scene=null}
    state.prediction=null;state.fitted=false;
  }
  function disposeAvatarContext(){
    const context=state.avatarContext;if(!context)return;
    context.outer.removeFromParent();disposeAvatarObject(context.container);state.avatarContext=null;
  }
  function renderError(error){state.loading=false;state.error=error?.message||"Results unavailable";render()}
  function renderEmpty(){state.record=null;state.loading=false;render()}
  function sourceSwitch(){const group=create("nav","results-source-switch");group.setAttribute("aria-label","Result source");for(const [source,label] of [["user","My evaluation"],["official","Official evaluation"]]){const button=actionButton(label,"source","results-source-button");button.dataset.resultSource=source;button.setAttribute("aria-pressed",String(source===state.source));group.append(button)}group.append(createHelpIcon("My evaluation contains your tests; Official evaluation contains the recorded paper tests.","Result source help"));return group}
  async function switchSource(source){if(!["user","official"].includes(source)||source===state.source)return;predictionDownload.select(null);state.request++;state.sampleRequest++;state.compareRequest++;disposeVisual();state.source=source;state.models=[];state.weights=[];state.records=[];state.model=null;state.weight=null;state.record=null;state.selected.clear();state.compareRecords=[];state.compareResult=null;state.baseline=null;state.compareError="";state.sequence="";state.samples=[];state.sampleTotal=0;await loadModels();if(state.view==="compare"){await loadCompareRecords();render()}}

  function openDeleteDialog(){
    if(state.deletePending||!canDeleteResult(state.record,state.source))return;
    state.deleteDialog?.close();const record=state.record,dialog=create("dialog","results-delete-dialog");state.deleteDialog=dialog;
    dialog.setAttribute("aria-labelledby","results-delete-title");
    const heading=create("header","results-delete-heading"),title=create("h2","","Delete evaluation");title.id="results-delete-title";heading.append(title);
    const body=create("div","results-delete-body");body.append(create("p","","Move this evaluation to local trash?"));
    const metadata=create("dl","results-delete-metadata");for(const [label,value] of [["Model",record.model],["Training run",record.training_run_id],["Test ID",record.test_id||record.record_id]])metadata.append(create("dt","",label),create("dd","",value||"—"));body.append(metadata);
    const error=create("p","results-delete-error");error.setAttribute("role","alert");error.hidden=true;body.append(error);
    const footer=create("footer","results-delete-actions"),cancel=actionButton("Cancel","cancel-delete","button"),confirm=actionButton("Delete evaluation","confirm-delete","button results-delete-button");footer.append(cancel,confirm);dialog.append(heading,body,footer);
    cancel.addEventListener("click",()=>{if(!state.deletePending)dialog.close()});confirm.addEventListener("click",()=>withLoading(()=>confirmDeletion(dialog,record)));
    dialog.addEventListener("cancel",event=>{if(state.deletePending)event.preventDefault()});dialog.addEventListener("close",()=>{if(state.deleteDialog===dialog)state.deleteDialog=null;dialog.remove()},{once:true});document.body.append(dialog);dialog.showModal();cancel.focus();
  }
  async function confirmDeletion(dialog,record){
    if(state.deletePending||state.disposed||!dialog.open||!canDeleteResult(record,state.source)||state.record?.record_id!==record.record_id)return;
    state.deletePending=true;dialog.querySelectorAll("button").forEach(button=>button.disabled=true);const error=dialog.querySelector('[role="alert"]');error.hidden=true;
    try{
      const result=await api("/api/results/records/delete",{method:"POST",body:{source:"user",record_ids:[record.record_id]}});
      if(!list(result.deleted).includes(record.record_id))throw Error("Evaluation was not deleted");
      state.request++;state.sampleRequest++;state.compareRequest++;state.selected.delete(record.record_id);state.compareRecords=state.compareRecords.filter(item=>item.record_id!==record.record_id);state.compareResult=null;if(state.baseline===record.record_id)state.baseline=null;
      const index=state.records.findIndex(item=>item.record_id===record.record_id);state.records=state.records.filter(item=>item.record_id!==record.record_id);state.record=state.records[Math.min(Math.max(index,0),state.records.length-1)]||null;state.samples=[];state.lastPayload=null;dialog.close();
      if(!state.disposed)await loadModels();
    }catch(failure){if(dialog.open){error.textContent=failure.message||"Unable to delete evaluation";error.hidden=false}}
    finally{state.deletePending=false;dialog.querySelectorAll("button").forEach(button=>button.disabled=false)}
  }

  function sectionHead(title,meta=""){
    const head=create("div","results-panel-head"),group=create("div","ui-title-group");group.append(create("h2","",title));
    const help={"Test details":"Inspect joint errors, saved files and parameters for the selected test.","Per-joint MLE":"Compare mean position errors for individual joints, shown in centimeters.","Visualization":"Inspect radar inputs alongside predicted and reference poses for the selected frame.","Completion metrics":"Inspect the saved Chamfer distance for the point completion task.","Test runs":"Select test records to compare their saved metrics.","Overall performance":"Compare saved metrics across selected tests without running new inference.","Shared samples":"Open synchronized views when selected tests share compatible sample identifiers.","MLE · cm":"Compare mean localization errors; lower values indicate smaller position errors."}[title];
    if(help)group.append(createHelpIcon(help,title+" help"));head.append(group);if(meta)head.append(create("small","",meta));return head;
  }
  function disposeSelectionDrawer(){
    clearTimeout(selectionDrawerOpenTimer);selectionDrawerOpenTimer=0;
    const current=selectionDrawer;selectionDrawer=null;
    if(current?.dialog.open)current.dialog.close();
  }
  function resizeResultsVisuals(){requestAnimationFrame(()=>{state.scene?.resize();state.diagram?.setActive(state.active)})}
  function setSelectionDrawerOpen(open){
    clearTimeout(selectionDrawerOpenTimer);selectionDrawerOpenTimer=0;
    if(!selectionDrawer||!mobileResultsQuery.matches)return;
    state.selectionDrawerOpen=Boolean(open);selectionDrawer.toggle.setAttribute("aria-expanded",String(state.selectionDrawerOpen));
    if(state.selectionDrawerOpen){selectionDrawer.dialog.classList.remove("restore-open");if(!selectionDrawer.dialog.open)selectionDrawer.dialog.showModal()}
    else if(selectionDrawer.dialog.open)selectionDrawer.dialog.close();
    resizeResultsVisuals();
  }
  function syncSelectionDrawer(){
    const current=selectionDrawer;if(!current)return;
    if(mobileResultsQuery.matches){current.body.append(current.controls);if(current.rail)current.body.append(current.rail);if(state.selectionDrawerOpen&&!current.dialog.open)current.dialog.showModal()}
    else{state.selectionDrawerOpen=false;if(current.dialog.open)current.dialog.close();current.home.append(current.controls);if(current.rail)current.railHome.append(current.rail)}
    current.toggle.setAttribute("aria-expanded",String(state.selectionDrawerOpen));resizeResultsVisuals();
  }
  function mountSelectionDrawer(page,{restoreOpen=false}={}){
    const home=page.querySelector(".results-selection-home"),controls=page.querySelector(".results-selection-controls");if(!home||!controls)return;
    const toggle=actionButton("","selection-drawer","results-mobile-drawer-toggle");toggle.setAttribute("aria-label",state.view==="compare"?"Select comparison runs":"Select model and test");toggle.setAttribute("aria-haspopup","dialog");toggle.setAttribute("aria-expanded",String(state.selectionDrawerOpen));toggle.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>';
    const dialog=create("dialog","results-mobile-drawer");if(restoreOpen)dialog.classList.add("restore-open");const heading=create("div","results-mobile-drawer-head"),title=create("h2","",state.view==="compare"?"Test runs":"Model and test"),close=actionButton("×","close-selection-drawer","results-mobile-drawer-close"),body=create("div","results-mobile-drawer-body");close.setAttribute("aria-label","Close selections");const drawerId=(root.id||"results-"+state.view)+"-mobile-selections";dialog.id=drawerId;title.id=drawerId+"-title";dialog.setAttribute("aria-labelledby",title.id);toggle.setAttribute("aria-controls",drawerId);heading.append(title,close);dialog.append(heading,body);page.append(toggle,dialog);
    const rail=page.querySelector(".results-compare-rail");let railHome=null;if(rail){railHome=create("div","results-compare-rail-home");rail.before(railHome);railHome.append(rail)}
    selectionDrawer={dialog,toggle,body,home,controls,rail,railHome};
    dialog.addEventListener("close",()=>{if(selectionDrawer?.dialog!==dialog)return;state.selectionDrawerOpen=false;toggle.setAttribute("aria-expanded","false");resizeResultsVisuals()});
    dialog.addEventListener("click",event=>{if(event.target!==dialog)return;const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)setSelectionDrawerOpen(false)});
    syncSelectionDrawer();
  }
  function render(){
    const restoreSelectionDrawer=Boolean(state.selectionDrawerOpen&&selectionDrawer?.dialog.open);disposeSelectionDrawer();disposeVisual();root.innerHTML="";const page=create("div","results-page"+(state.view==="compare"?" results-comparison-page":"")),header=create("header","results-header"),title=create("div","results-title-block");
    const heading=create("div","ui-title-group");heading.append(create("h1","results-heading",state.view==="compare"?"Compare runs":"Results"),createHelpIcon(state.view==="compare"?"Compare saved evaluation metrics across selected tests without running new inference.":"Browse your tests or recorded official evaluations, including metrics and available predictions.",state.view==="compare"?"Compare runs help":"Results help"));
    title.append(create("div","results-eyebrow",state.view==="compare"?"03 / COMPARISON":"03 / RESULTS"),heading);header.append(title);
    const actions=create("div","results-header-actions");actions.append(state.view==="compare"?actionButton("← Results","back","results-back-link"):actionButton("Compare runs →","compare","button button-primary"));actions.prepend(actionButton("Refresh","refresh-results","button button-quiet"));if(state.view!=="compare"&&!state.loading&&!state.error&&canDeleteResult(state.record,state.source)){const remove=actionButton("Delete test","delete-test","button results-delete-button");remove.disabled=state.deletePending;actions.append(remove)}header.append(actions);const selectionHome=create("div","results-selection-home"),selectionControls=create("div","results-selection-controls");selectionControls.append(sourceSwitch());selectionHome.append(selectionControls);page.append(header,selectionHome);
    if(state.loading||state.error||!state.models.length){page.append(create("div",state.error?"results-error":"results-state",state.loading?"Loading":state.error||"No test runs"));root.append(page);mountSelectionDrawer(page,{restoreOpen:restoreSelectionDrawer});return}
    if(state.view==="compare"){renderCompare(page);root.append(page);mountSelectionDrawer(page,{restoreOpen:restoreSelectionDrawer});return}
    const selectors=create("section","results-selectors");selectors.append(selectField("Model","model",state.models,state.model,x=>x.name||x.id),selectField("Weights","weight",state.weights,state.weight,x=>(x.training_run_id||x.id)+" / "+String(x.checkpoint||"best.ckpt").split("/").at(-1)),selectField("Test ID","record",state.records,state.record?.record_id,x=>(x.test_id||x.record_id)+(x.mtime?" · "+new Date(x.mtime*1000).toLocaleString("en-GB",{hour12:false}):"")));
    const status=create("div","results-record-state");if(state.record){status.append(badge(state.record.status||"Unknown",state.record.complete?"is-good":"is-warning"),badge(hasPkl()?"Prediction PKL":"Metrics"));const tables=paperTableTags({...state.record,origin:state.record.source==='official'?'official':'user'});if(tables.length)status.classList.add('has-paper-tags');for(const tag of tables)status.append(badge(tag.label,'is-paper-table'))}selectors.append(status);selectionControls.append(selectors);
    if(!state.record){page.append(create("div","results-state","No test records"));root.append(page);mountSelectionDrawer(page,{restoreOpen:restoreSelectionDrawer});return}
    if(state.source==='official'&&state.record.source==='official'&&officialAssetId(state.record)){const transfer=create('div','results-prediction-download');transfer.dataset.officialPredictionDownload='';transfer.innerHTML=predictionDownloadHtml();page.append(transfer)}
    const record=state.record,m=record.metrics||{},stats=create("section","results-metrics");if(isCompletion()){stats.append(metricCard("Chamfer distance ↓",m.cd_cm,"cm"),metricCard("Samples",record.samples,""))}else stats.append(metricCard("MLE ↓",m.mle_cm,"cm"),metricCard("PA-MPJPE ↓",m.pa_mpjpe_cm,"cm"),metricCard(record.threshold_m?"PCK@"+(Number(record.threshold_m)*100).toFixed(0)+"cm ↑":"PCK ↑",m.pck_percent,"%"),metricCard("AUC ↑",m.auc_percent,"%"));page.append(stats);
    if(hasPkl()&&!state.predictionNeedsOpen)renderVisualization(page);else renderMetricDetails(page);root.append(page);mountSelectionDrawer(page,{restoreOpen:restoreSelectionDrawer});
    const diagram=root.querySelector("[data-joint-diagram-mount]");if(diagram){state.diagram=mountJointDiagram(diagram,{rows:jointRows(state.record.per_joint)});state.diagram.setActive(state.active)}
  }
  function selectField(label,key,items,value,format){
    const wrap=create("label","results-field");wrap.append(create("span","results-field-label",label));const select=create("select","results-select");select.dataset.select=key;
    const tagged=key==="weight"||key==="record";
    if(tagged){select.classList.add("results-table-select");const button=create("button","results-selected-button");button.type="button";button.append(create("selectedcontent"));select.append(button)}
    for(const item of items){
      const text=format(item),option=create("option");option.value=item.id||item.record_id;option.selected=option.value===value;
      if(tagged){
        const tags=paperTableTags({...item,origin:state.source==="official"&&item.source!=="user"?"official":"user"});option.append(create("span","results-option-label",text));
        if(tags.length){const group=create("span","results-option-tags");for(const tag of tags){const chip=badge(tag.label,"is-paper-table"+(tag.dependency?" is-dependency":""));group.append(chip)}option.append(document.createTextNode(" "),group)}
        option.title=[text,...tags.map(tag=>tag.label)].join(" · ");option.setAttribute("aria-label",option.title);
      }else option.textContent=text;
      select.append(option);
    }
    wrap.append(select);return wrap;
  }
  function detailTabs(){const tabs=create("nav","results-detail-tabs");for(const key of ["joints","artifacts","run"]){const b=actionButton(key==="joints"?(isCompletion()?"Summary":"Joints"):key==="artifacts"?"Artifacts":"Run","detail");b.dataset.detail=key;if(key===state.detail)b.classList.add("is-active");tabs.append(b)}return tabs}
  function detailContent(){const content=create("div","results-detail-content");if(state.detail==="joints")content.append(isCompletion()?configTable(state.record):perJointTable(state.record.per_joint));else if(state.detail==="artifacts")content.append(artifactTable(state.record));else content.append(configTable(state.record));return content}
  function detailPanel(){const detail=create("aside","results-details");detail.append(sectionHead("Test details",state.record.joints?`${state.record.joints} joints`:""),detailTabs(),detailContent());if(hasPkl()){const sample=currentSample(),footer=create("div","results-sample-footer");footer.append(create("label","","Sequence"),create("label","","Target frame"));const selector=create("select","results-sequence-id");selector.dataset.sequenceSelect="";selector.setAttribute("aria-label","Sequence");const all=create("option","","All sequences");all.value="";selector.append(all);for(const id of state.sequenceIds){const option=create("option","",id);option.value=id;option.selected=id===state.sequence;selector.append(option)}selector.value=state.sequence;const target=create("input","results-target-frame");target.type="number";target.min="0";target.step="1";target.value=sample?.target_frame??"";target.dataset.targetFrame="";target.setAttribute("aria-label","Target frame");footer.append(selector,target);detail.append(footer)}return detail}
  function jointRows(data){return (Array.isArray(data)?data:Object.entries(data||{}).map(([joint,values])=>({joint,...(values||{})}))).map((item,index)=>({...item,index:item.index??item.joint_index??index,label:DISPLAY_JOINTS[item.index??item.joint_index]||item.joint||item.name||`Joint ${index+1}`,value:typeof item.mle_cm==="number"&&Number.isFinite(item.mle_cm)?item.mle_cm:null}))}
  function perJointTable(data){const wrap=create("div","results-joints-view"),rows=jointRows(data),max=Math.max(...rows.map(row=>row.value||0),1);for(const item of rows){const row=create("div","results-joint-row"),track=create("div","results-bar-track"),fill=create("div","results-bar-fill");fill.style.width=(item.value===null?0:Math.min(100,item.value/max*80))+"%";track.append(fill);row.append(create("span","",item.label),track,create("code","",fmt(item.value)));wrap.append(row)}if(!rows.length)wrap.append(create("div","results-state","No per-joint metrics"));return wrap}
  function metricPanel(){const aggregate=create("section","results-viewer results-aggregate results-error-panel");aggregate.append(sectionHead(isCompletion()?"Completion metrics":"Per-joint MLE","cm"));const chart=create("div","results-aggregate-chart"+(isCompletion()?"":" results-joint-diagram-mount"));if(isCompletion()){const summary=create("div","results-completion-summary");summary.append(create("span","results-metric-label","Chamfer distance"),create("strong","results-metric-value",fmt(state.record.metrics?.cd_cm)),create("small","results-metric-unit","cm"));chart.append(summary)}else if(jointRows(state.record.per_joint).length)chart.dataset.jointDiagramMount="";else chart.append(create("div","results-state","No per-joint metrics"));aggregate.append(chart);if(!isCompletion()&&state.predictionLoadingId===state.record.record_id)aggregate.append(create("div","results-state","Loading prediction…"));else if(!isCompletion()&&state.predictionErrorId===state.record.record_id)aggregate.append(create("div","results-state","Prediction unavailable; metrics remain available."));return aggregate}
  function renderMetricDetails(page){const body=create("div","results-visual-layout results-metrics-layout");body.append(metricPanel(),detailPanel());page.append(body)}
  function renderVisualization(page){
    const dual=Boolean(state.record?.per_joint&&!isCompletion()&&jointRows(state.record.per_joint).length);
    const layout=create("section","results-visual-layout"+(dual?" results-dual-layout":"")),viewer=create("div","results-viewer results-prediction-panel");if(dual)layout.append(metricPanel());viewer.append(sectionHead("Visualization","XYZ · m"));
    const viewport=create("div","results-canvas-wrap"),canvas=create("div","results-canvas");canvas.id="results-spatial";viewport.append(canvas,create("span","results-canvas-status","Select a sample"));viewport.lastChild.dataset.canvasStatus="";viewer.append(viewport);
    const controls=create("div","results-view-controls");for(const [key,label] of [["points","Radar"],["gt","Ground truth"],["predictionVisible","Prediction"]]){const toggle=create("label","results-toggle"),input=create("input");input.type="checkbox";input.dataset.toggle=key;input.checked=Boolean(state[key]);toggle.append(input,create("span","",label));controls.append(toggle)}
    if(state.avatarModels.length){
      const group=create("div","results-avatar-controls");group.setAttribute("role","group");group.setAttribute("aria-label","Avatar settings");
      const toggle=create("label","results-toggle"),input=create("input");input.type="checkbox";input.dataset.toggle="avatarVisible";input.checked=state.avatarVisible;toggle.append(input,create("span","","Avatar"));group.append(toggle,create("span","results-badge","Xbot"));
      const sourceWrap=create("label","results-avatar-source-label","Source"),sourceSelect=create("select","results-avatar-select");sourceSelect.dataset.avatarSource="";for(const [value,label] of [["gt","Ground truth"],["prediction","Prediction"]]){const option=create("option","",label);option.value=value;option.selected=value===state.avatarSource;sourceSelect.append(option)}sourceWrap.append(sourceSelect);group.append(sourceWrap);
      const opacity=create("input","results-avatar-opacity");opacity.type="range";opacity.min="0.15";opacity.max="1";opacity.step="0.05";opacity.value=state.avatarOpacity;opacity.title="Avatar opacity";opacity.setAttribute("aria-label","Avatar opacity");opacity.dataset.avatarOpacity="";group.append(opacity);controls.append(group);
    }
    controls.append(actionButton("Reset","reset-view","button button-quiet"));viewer.append(controls);const status=create("span","results-avatar-status",state.avatarStatus);status.dataset.avatarStatus="";viewer.append(status);
    const scrub=create("div","results-scrubber"),range=create("input");range.type="range";range.min="0";range.max=Math.max(0,state.sampleTotal-1);range.value=state.sampleIndex;range.dataset.scrubber="";range.setAttribute("aria-label","Sample timeline");range.disabled=!state.samples.length;scrub.append(actionButton("⇤","first-sample","button button-quiet"),actionButton("←","prev-sample","button button-quiet"),actionButton(state.playing?"Pause":"▶ Play","play-samples","button button-primary"),actionButton("→","next-sample","button button-quiet"),actionButton("⇥","last-sample","button button-quiet"),range,create("span","results-sample-count",state.samples.length?`${state.sampleIndex+1} / ${state.sampleTotal}`:"0 / 0"));scrub.dataset.loading="false";viewer.append(scrub);
    layout.append(viewer,detailPanel());page.append(layout);
    requestAnimationFrame(()=>{if(state.disposed||!root.contains(canvas)||canvas.querySelector("canvas"))return;try{state.scene=new SpatialView(canvas);state.scene.setOptions({radar:state.points,skeleton:state.gt,boundary:false});state.scene.fit();if(state.lastPayload)drawFrame(state.lastPayload)}catch(_error){setCanvasStatus("Visualization unavailable")}})
  }
  function artifactTable(record){const wrap=create("div","results-artifacts");for(const [name,value,status] of [["Prediction PKL",record.prediction_pkl?.path,hasPkl()?"Available":record.prediction_pkl?.present?"Unavailable":"Missing"],["Numeric predictions",record.prediction_npz?.manifest,record.prediction_npz?.present?"Available":"Missing"],[record.split_strategy?"Split strategy":"Split manifest",record.split_strategy||record.split_manifest,(record.split_strategy||record.split_manifest)?"Available":"Missing"],...(record.partition_cache_id?[["Partition cache ID",record.partition_cache_id,"Available"]]:[]),["Dataset manifest SHA-256",record.dataset_manifest_sha256,""],["Split index SHA-256",record.split_index_sha256,""]]){const row=create("div","results-artifact-row");row.append(create("span","",name),create("code","",value||"-"),create("span","",status));wrap.append(row)}return wrap}
  function configTable(record){const wrap=create("div","results-config");const rows=[["Record ID",record.record_id],["Model",record.model],["Training run",record.training_run_id],["Weight",record.weight_id],["Test ID",record.test_id],["Action",record.action],["Status",record.status],["Split",record.split],["Threshold",record.threshold_m],["Seed",record.seed],["Visual requested",record.visual_requested],["Created",record.mtime],...Object.entries(record.config||{})];for(const [key,value] of rows){const row=create("div","results-config-row");row.append(create("span","",key),create("strong","",typeof value==="object"?JSON.stringify(value):value??"-"));wrap.append(row)}return wrap}
  function setCanvasStatus(message){const node=root.querySelector("[data-canvas-status]");if(node)node.textContent=message}
  function drawFrame(payload){
    state.lastPayload=payload;if(!state.scene)return;
    const scrubber=root.querySelector("[data-scrubber]"),counter=root.querySelector(".results-sample-count"),sequence=root.querySelector(".results-sequence-id"),frameLabel=root.querySelector(".results-target-frame");if(scrubber)scrubber.value=state.sampleIndex;if(counter)counter.textContent=`${state.sampleIndex+1} / ${state.sampleTotal}`;if(sequence)sequence.value=state.sequence;if(frameLabel)frameLabel.value=payload.target_frame??"";
    const joints=list(payload.target).map(point=>finite(point)?point.slice(0,3):null);
    const frame={radar:list(payload.radar),joints,edges:BONES,index:state.sampleIndex,generation:state.sampleIndex};
    state.scene.setFrame(frame,{sequence:payload.sequence_id,sample:payload.sample_id});
    if(!state.fitted){state.scene.fit();state.fitted=true}
    if(!state.prediction){state.prediction=makePrediction();state.scene.scene.add(state.prediction)}
    updatePrediction(state.prediction,payload.prediction);state.prediction.visible=state.predictionVisible;
    state.scene.renderer.domElement.dataset.predictionJoints=String(state.prediction?.userData.jointCount||0);
    if(state.avatarVisible)ensureAvatar(payload.avatar_pose?.[state.avatarSource]);else if(state.avatarContext)state.avatarContext.outer.visible=false;
    state.scene.render();setCanvasStatus("");
  }
  function makePrediction(){
    const group=new THREE.Group(),style=PAGES_RENDER_CONFIG.keypoints;
    group.name="prediction";
    const geometry=new THREE.BufferGeometry(),texture=createGlowTexture(style.glowTextureSize);
    const material=new THREE.PointsMaterial({color:0xffbd4a,size:style.size,map:texture,alphaMap:texture,transparent:true,opacity:style.mainOpacity,alphaTest:style.mainAlphaTest,depthWrite:false,sizeAttenuation:true});
    const joints=new THREE.Points(geometry,material);joints.name="prediction-joints";joints.frustumCulled=false;
    const haloMaterial=new THREE.PointsMaterial({color:0xffbd4a,size:style.size*style.glowScale,map:texture,alphaMap:texture,transparent:true,opacity:style.glowOpacity,alphaTest:style.glowAlphaTest,depthWrite:false,blending:THREE.AdditiveBlending,sizeAttenuation:true});
    const halo=new THREE.Points(geometry,haloMaterial);halo.name="prediction-joint-glow";halo.frustumCulled=false;group.add(halo,joints);
    const bones=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffbd4a,transparent:true,opacity:.95}));bones.name="prediction-bones";bones.frustumCulled=false;group.add(bones);
    return group;
  }
  function updatePrediction(group,points){
    const valid=list(points).map(p=>finite(p)?p:null),positions=valid.filter(Boolean).flatMap(p=>p.slice(0,3)),lines=[];
    for(const [a,b] of BONES)if(valid[a]&&valid[b])lines.push(...valid[a].slice(0,3),...valid[b].slice(0,3));
    for(const [name,values] of [["prediction-joints",positions],["prediction-bones",lines]]){
      const geometry=group.getObjectByName(name).geometry;let attribute=geometry.getAttribute("position");
      if(!attribute||attribute.array.length<values.length){attribute=new THREE.BufferAttribute(new Float32Array(Math.max(values.length,JOINTS.length*6)),3);attribute.setUsage(THREE.DynamicDrawUsage);geometry.setAttribute("position",attribute)}
      attribute.array.set(values);attribute.needsUpdate=true;geometry.setDrawRange(0,values.length/3);
    }
    group.userData.jointCount=positions.length/3;
  }
  function parseDirection(value){if(!Array.isArray(value)||value.length<3)return null;const vector=new THREE.Vector3(Number(value[0]),Number(value[1]),Number(value[2]));return vector.toArray().every(Number.isFinite)&&vector.lengthSq()>1e-10?vector.normalize():null}
  function avatarRootRotation(pose){const directions=pose.target_directions||{},left=parseDirection(directions.left_shoulder),right=parseDirection(directions.right_shoulder),up=parseDirection(directions.upper_spine);if(!left||!right||!up)return new THREE.Quaternion();const z=up.clone(),x=left.clone().sub(right).addScaledVector(z,-left.clone().sub(right).dot(z));if(x.lengthSq()<1e-10)return new THREE.Quaternion();x.normalize();const y=z.clone().cross(x);if(y.lengthSq()<1e-10)return new THREE.Quaternion();y.normalize();x.copy(y).cross(z).normalize();return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z)).normalize()}
  function applyAvatarOpacity(context){for(const mesh of context.skinnedMeshes)for(const material of (Array.isArray(mesh.material)?mesh.material:[mesh.material]))if(material){if(material.userData.resultsBaseOpacity===undefined){material.userData.resultsBaseOpacity=material.opacity;material.userData.resultsBaseTransparent=material.transparent;material.userData.resultsBaseDepthWrite=material.depthWrite}material.transparent=state.avatarOpacity<.999||material.userData.resultsBaseTransparent;material.opacity=material.userData.resultsBaseOpacity*state.avatarOpacity;material.depthWrite=state.avatarOpacity<.999?false:material.userData.resultsBaseDepthWrite}}
  function buildAvatarContext(gltf,descriptor){
    const source=gltf?.scene;if(!source)throw new Error("GLB scene unavailable");const profile=descriptor.rig_profile||{};if(!profile.root_bone||!Array.isArray(profile.bindings)||!profile.bindings.length)throw new Error("Rig profile unavailable");
    const outer=new THREE.Group(),root=new THREE.Group(),container=new THREE.Group();outer.add(root);root.add(container);const euler=profile.alignment_euler_deg||[0,0,0];container.rotation.set(...euler.slice(0,3).map(THREE.MathUtils.degToRad),profile.alignment_order||"XYZ");source.scale.setScalar(Number(profile.unit_scale)||1);
    const bones=new Map(),skinnedMeshes=[];source.traverse(object=>{if(object.isBone){if(object.name)bones.set(object.name,object);if(object.userData?.name)bones.set(object.userData.name,object)}if(object.isSkinnedMesh){object.frustumCulled=false;skinnedMeshes.push(object)}});
    const rootBone=bones.get(profile.root_bone);if(!rootBone||!skinnedMeshes.length)throw new Error("Skinned rig bones unavailable");const rootPosition=rootBone.getWorldPosition(new THREE.Vector3());source.position.sub(rootPosition);container.add(source);container.updateMatrixWorld(true);
    const bindQuaternions=new Map(),bindings=[];for(const tuple of profile.bindings.concat(profile.optional_bindings||[])){if(!Array.isArray(tuple)||tuple.length<3)continue;const [semantic,boneName,childName]=tuple,bone=bones.get(boneName),child=bones.get(childName);if(!bone||!child){if((profile.bindings||[]).includes(tuple))throw new Error(`Rig binding unavailable: ${boneName}`);continue}if(!bindQuaternions.has(bone))bindQuaternions.set(bone,bone.quaternion.clone());bindings.push({semantic,bone,child})}
    const baseHeight=new THREE.Box3().setFromObject(container).getSize(new THREE.Vector3()).z;if(!Number.isFinite(baseHeight)||baseHeight<=1e-6)throw new Error("Avatar dimensions unavailable");
    const context={outer,root,container,bones,skinnedMeshes,bindQuaternions,bindings,profile,baseHeight};applyAvatarOpacity(context);return context;
  }
  async function loadAvatarModel(modelId){
    if(state.avatarContext?.modelId===modelId)return state.avatarContext;
    if(state.avatarLoadingId===modelId&&state.avatarLoadingPromise)return state.avatarLoadingPromise;
    const descriptor=state.avatarModels.find(item=>item.id===modelId);if(!descriptor||!descriptor.asset_url)throw new Error("Avatar asset unavailable");const generation=++state.avatarRequest;disposeAvatarContext();state.avatarStatus="Loading";updateAvatarStatus();state.avatarLoadingId=modelId;
    const pending=withLoading(async()=>{if(generation!==state.avatarRequest)return null;const asset=await api(descriptor.asset_url,{image:true});const buffer=await asset.arrayBuffer();const loader=new GLTFLoader();const gltf=await loader.parseAsync(buffer,"");if(generation!==state.avatarRequest){disposeGltf(gltf);return null}let context;try{context=buildAvatarContext(gltf,descriptor)}catch(error){disposeAvatarObject(gltf.scene);disposeGltf(gltf);throw error}if(generation!==state.avatarRequest){disposeAvatarObject(context.container);return null}context.modelId=modelId;state.avatarContext=context;state.scene?.scene.add(context.outer);state.avatarStatus="";updateAvatarStatus();return context},{delay:0});state.avatarLoadingPromise=pending;
    try{return await pending}finally{if(generation===state.avatarRequest){state.avatarLoadingId=null;state.avatarLoadingPromise=null}}
  }
  function disposeGltf(gltf){if(gltf?.scene)disposeAvatarObject(gltf.scene)}
  function disposeAvatarObject(object){if(!object)return;const geometries=new Set(),materials=new Set(),textures=new Set(),skeletons=new Set();object.traverse(node=>{if(node.geometry)geometries.add(node.geometry);if(node.skeleton)skeletons.add(node.skeleton);for(const material of (Array.isArray(node.material)?node.material:[node.material]))if(material){materials.add(material);for(const value of Object.values(material))if(value?.isTexture)textures.add(value)}});textures.forEach(value=>value.dispose());materials.forEach(value=>value.dispose());geometries.forEach(value=>value.dispose());skeletons.forEach(value=>value.dispose?.())}
  async function ensureAvatar(pose){
    if(!state.avatarVisible||!state.avatarModelId||!state.scene)return;
    if(!pose||pose.protocol!=="avatar_pose/v1"||pose.valid===false||pose.coordinate_frame!=="world_z_up"){if(state.avatarContext)state.avatarContext.outer.visible=false;state.avatarStatus="Pose unavailable";updateAvatarStatus();return}
    try{let context=state.avatarContext;if(!context||context.modelId!==state.avatarModelId)context=await loadAvatarModel(state.avatarModelId);if(!context)return;if(!state.avatarVisible){context.outer.visible=false;return}const currentPose=state.lastPayload?.avatar_pose?.[state.avatarSource];if(!currentPose||currentPose.valid===false){context.outer.visible=false;return}applyAvatarPose(context,currentPose);context.outer.visible=true;state.avatarStatus="";updateAvatarStatus();state.scene?.render()}catch(error){state.avatarStatus=error?.message||"Avatar unavailable";updateAvatarStatus();if(state.avatarContext)state.avatarContext.outer.visible=false}
  }
  function applyAvatarPose(context,pose){
    const rootPosition=pose.root_position;if(!Array.isArray(rootPosition)||rootPosition.length<3||!rootPosition.every(value=>Number.isFinite(Number(value))))throw new Error("Avatar pose unavailable");
    for(const [bone,quaternion] of context.bindQuaternions)bone.quaternion.copy(quaternion);
    context.root.position.fromArray(rootPosition);context.root.quaternion.copy(avatarRootRotation(pose));const bodyHeight=Number(pose.target_height_m),poseScale=Number(pose.scale)||1;context.root.scale.setScalar(bodyHeight>0?THREE.MathUtils.clamp(bodyHeight/context.baseHeight,.1,5):THREE.MathUtils.clamp(poseScale,.1,5));context.root.updateMatrixWorld(true);
    for(const binding of context.bindings){const target=parseDirection(pose.target_directions?.[binding.semantic]);if(!target)continue;const start=binding.bone.getWorldPosition(new THREE.Vector3()),end=binding.child.getWorldPosition(new THREE.Vector3()),direction=end.sub(start);if(direction.lengthSq()<1e-10)continue;direction.normalize();const delta=new THREE.Quaternion().setFromUnitVectors(direction,target),current=binding.bone.getWorldQuaternion(new THREE.Quaternion()),desired=delta.multiply(current).normalize(),parentInverse=binding.bone.parent?binding.bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert():new THREE.Quaternion();binding.bone.quaternion.copy(parentInverse.multiply(desired).normalize());binding.bone.updateWorldMatrix(true,true)}
    const footLevel=Number(pose.foot_level_z),footBones=(context.profile.foot_bones||[]).map(name=>context.bones.get(name)).filter(Boolean);if(Number.isFinite(footLevel)&&footBones.length){const modelLevel=Math.min(...footBones.map(bone=>bone.getWorldPosition(new THREE.Vector3()).z));context.root.position.z+=footLevel-modelLevel;context.root.updateMatrixWorld(true)}
    applyAvatarOpacity(context);
  }
  function updateAvatarStatus(){const node=root.querySelector("[data-avatar-status]");if(node)node.textContent=state.avatarStatus}
  function updatePlay(){const button=root.querySelector('[data-action="play-samples"]');if(button)button.textContent=state.playing?"Pause":"Play"}
  async function loadCompareRecords(){
    const rows=[],source=state.source,request=state.compareRequest;
    for(const model of state.models){
      try{
        const payload=await resultsApi("/api/results/models/"+encodeURIComponent(model.id)+"/weights");
        for(const weight of list(payload.weights))for(const test of list(weight.tests))rows.push({...test,model:model.id,weight_id:weight.id});
      }catch(_error){}
    }
    if(source===state.source&&request===state.compareRequest)state.compareRecords=rows;
  }
  function comparisonColor(record){return compareColors[Math.max(0,state.compareRecords.findIndex(item=>item.record_id===record.record_id))%compareColors.length]}
  function comparePanel(title,content,meta=""){const panel=create("section","results-compare-panel");panel.append(sectionHead(title,meta),content);return panel}
  function renderCompare(page){
    const grid=create("div","results-compare-grid"),rail=create("aside","results-compare-rail"),railHead=sectionHead("Test runs");const selectedCount=badge(`${state.selected.size} selected`);selectedCount.dataset.compareCount="";railHead.append(selectedCount);rail.append(railHead);const runList=create("div","results-compare-list");
    state.compareRecords.forEach((record,index)=>{const row=create("label","results-compare-run"+(state.selected.has(record.record_id)?" is-selected":"")),check=create("input");check.type="checkbox";check.dataset.compareRecord=record.record_id;check.dataset.loading="false";check.checked=state.selected.has(record.record_id);const copy=create("span","results-compare-run-copy"),name=create("strong","",record.model||record.test_id);name.style.color=comparisonColor(record);copy.append(name,create("code","",record.test_id||record.record_id),create("small","",String(record.status||"unknown")));const tables=create("span","results-compare-table-tags");for(const tag of paperTableTags({...record,origin:record.source==="official"?"official":"user"}))tables.append(badge(tag.label,"is-paper-table"));if(tables.childNodes.length)copy.append(tables);row.append(check,copy);runList.append(row)});rail.append(runList);
    grid.append(rail);grid.append(comparisonContent());page.append(grid);
  }
  function comparisonContent(){
    const content=create("div","results-compare-content"),result=state.compareResult,records=list(result?.records).slice().sort((a,b)=>a.record_id===state.baseline?-1:b.record_id===state.baseline?1:0);if(state.compareError)content.append(create("div","results-error",state.compareError));
    content.append(comparePanel("Overall performance",comparisonTable(records,Boolean(result?.compatible))));const charts=create("div","results-compare-charts");charts.append(comparePanel("MLE · cm",comparisonChart(records,"mle_cm")),comparePanel("Per-joint MLE",comparisonJoints(records)));content.append(charts);
    const shared=create("div","results-shared-body"),facts=create("dl","results-shared-facts"),count=Number(result?.shared_samples)||0,total=Math.max(...records.map(record=>Number(record.samples)||0),0);for(const [label,value] of [["Samples",count.toLocaleString()],["Coverage",total?`${(count/total*100).toFixed(1)}%`:"—"],["Alignment","Sequence / Target frame"]])facts.append(create("dt","",label),create("dd","",value));shared.append(facts);const controls=create("div","results-sync-view"),syncButton=actionButton("Synchronized view","sync-view","button button-quiet");syncButton.disabled=!result?.synchronization_allowed;if(result?.synchronization_allowed){const select=create("select","results-select");select.dataset.syncSample="";select.setAttribute("aria-label","Shared sample");for(const id of list(result.sample_ids)){const option=create("option","",id);option.value=id;select.append(option)}controls.append(select)}controls.append(syncButton);shared.append(controls);content.append(comparePanel("Shared samples",shared));return content;
  }
  function comparisonTable(records,compatible){const wrap=create("div","results-table-wrap results-comparison-table-wrap"),table=create("table","results-table results-comparison-table"),head=create("thead"),row=create("tr");["Model / Test","MLE ↓ cm","PA-MPJPE ↓ cm","PCK ↑ %","AUC ↑ %","Δ MLE"].forEach(value=>row.append(create("th","",value)));head.append(row);table.append(head);const body=create("tbody"),baseline=records[0]?.metrics?.mle_cm;for(const [index,record] of records.entries()){const tr=create("tr"),name=create("td","",`${record.model} / ${record.test_id}`);name.style.color=comparisonColor(record);tr.append(name);for(const key of ["mle_cm","pa_mpjpe_cm","pck_percent","auc_percent"]){const values=records.map(item=>item.metrics?.[key]).filter(value=>typeof value==="number"&&Number.isFinite(value)),value=record.metrics?.[key],best=values.length?(key==="pck_percent"||key==="auc_percent"?Math.max(...values):Math.min(...values)):null;tr.append(create("td",compatible&&typeof value==="number"&&value===best?"results-best":"",fmt(value)))}const value=record.metrics?.mle_cm;tr.append(create("td","",!compatible?"—":index===0?"Baseline":typeof value==="number"&&typeof baseline==="number"?`${value-baseline>0?"+":""}${fmt(value-baseline)} cm`:"—"));body.append(tr)}if(!records.length){const tr=create("tr"),td=create("td","results-empty-cell","Select two or more test records");td.colSpan=6;tr.append(td);body.append(tr)}table.append(body);wrap.append(table);return wrap}
  function comparisonChart(records,key){const valid=records.filter(record=>typeof record.metrics?.[key]==="number"&&Number.isFinite(record.metrics[key]));if(!valid.length)return create("div","results-state","No metric values");const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.classList.add("results-comparison-chart");svg.setAttribute("viewBox","0 0 500 200");svg.setAttribute("role","img");svg.setAttribute("aria-label","MLE comparison in centimeters");const max=Math.max(...valid.map(record=>record.metrics[key]),1)*1.15;let markup="";for(let i=0;i<=3;i++){const y=158-i*40;markup+=`<path class="results-chart-grid" d="M42 ${y}H480"/><text x="10" y="${y+4}">${fmt(max*i/3,1)}</text>`}valid.forEach((record,index)=>{const slot=430/valid.length,w=Math.min(64,slot*.62),x=48+index*slot+(slot-w)/2,h=record.metrics[key]/max*120;markup+=`<rect x="${x}" y="${158-h}" width="${w}" height="${h}" fill="${comparisonColor(record)}" opacity=".85"/><text class="results-chart-value" x="${x+w/2}" y="${150-h}" text-anchor="middle">${fmt(record.metrics[key])}</text><text x="${x+w/2}" y="183" text-anchor="middle">${esc(record.model)}</text>`});svg.innerHTML=markup;return svg}
  function comparisonJoints(records){const wrap=create("div","results-comparison-joints"),rows=records.map(record=>jointRows(record.per_joint)),indices=[...new Set(rows.flatMap(items=>items.map(item=>item.index)))].filter(index=>index!==undefined),maximum=Math.max(...rows.flatMap(items=>items.map(item=>item.value||0)),1);for(const index of indices){const row=create("div","results-comparison-joint-row"),bars=create("div","results-grouped-bars");row.append(create("span","",DISPLAY_JOINTS[index]||`Joint ${index}`));rows.forEach((items,recordIndex)=>{const item=items.find(value=>value.index===index),track=create("div","results-bar-track"),fill=create("div","results-bar-fill");fill.style.width=(item?.value===null||item?.value===undefined?0:item.value/maximum*90)+"%";fill.style.background=comparisonColor(records[recordIndex]);fill.title=`${records[recordIndex].model}: ${fmt(item?.value)} cm`;track.append(fill);bars.append(track)});row.append(bars);wrap.append(row)}if(!indices.length)wrap.append(create("div","results-state","No per-joint metrics"));return wrap}
  function updateComparisonSelection(){
    const count=root.querySelector('[data-compare-count]');if(count)count.textContent=`${state.selected.size} selected`;
    for(const input of root.querySelectorAll('[data-compare-record]')){
      const selected=state.selected.has(input.dataset.compareRecord);input.checked=selected;
      input.closest('.results-compare-run')?.classList.toggle('is-selected',selected);
    }
  }
  function updateComparison(){
    updateComparisonSelection();
    const previous=root.querySelector('.results-compare-content');if(!previous){render();return}
    const top=previous.scrollTop,jointTop=previous.querySelector('.results-comparison-joints')?.scrollTop||0;
    const next=comparisonContent();previous.replaceWith(next);next.scrollTop=top;
    const joints=next.querySelector('.results-comparison-joints');if(joints)joints.scrollTop=jointTop;
  }
  async function refreshComparison(){
    const ids=[...state.selected],request=++state.compareRequest;state.compareError='';
    if(!state.baseline||!state.selected.has(state.baseline))state.baseline=ids[0]||null;
    updateComparisonSelection();
    if(ids.length<2){state.compareResult=null;updateComparison();return}
    try{
      const result=await resultsApi('/api/results/compare',{method:'POST',body:{record_ids:ids}});
      if(request!==state.compareRequest||state.disposed)return;
      state.compareResult=result;updateComparison();
    }catch(error){
      if(request!==state.compareRequest||state.disposed)return;
      state.compareResult=null;state.compareError=error.message||'Comparison unavailable';updateComparison();
    }
  }
  function showDetail(detail){state.detail=detail;root.querySelectorAll("[data-detail]").forEach(button=>button.classList.toggle("is-active",button.dataset.detail===detail));const content=root.querySelector(".results-detail-content");if(content)content.replaceWith(detailContent())}
  async function onChange(event){
    const target=event.target;
    if(target.matches("[data-select=model]")){state.model=target.value;state.weight=null;state.record=null;state.compareResult=null;await loadWeights();return}
    if(target.matches("[data-select=weight]")){state.weight=target.value;state.record=null;state.compareResult=null;await loadTests();return}
    if(target.matches("[data-select=record]")){setSelectionDrawerOpen(false);state.record=state.records.find(r=>r.record_id===target.value)||null;state.compareResult=null;state.view="results";onViewChange("results");await loadRecord();return}
    if(target.matches("[data-toggle]")){const key=target.dataset.toggle;state[key]=target.checked;if(key==="avatarVisible"){if(!state.scene)return;if(state.avatarVisible&&state.lastPayload)await ensureAvatar(state.lastPayload.avatar_pose?.[state.avatarSource]);else if(state.avatarContext)state.avatarContext.outer.visible=false;state.scene.render()}else if(key==="predictionVisible"&&state.prediction){state.prediction.visible=state.predictionVisible;state.scene?.render()}else{state.scene?.setOptions({radar:state.points,skeleton:state.gt});state.scene?.render()}return}
    if(target.matches("[data-avatar-source]")){state.avatarSource=target.value;if(state.avatarVisible&&state.lastPayload)await ensureAvatar(state.lastPayload.avatar_pose?.[state.avatarSource]);return}
    if(target.matches("[data-sequence-select]")){state.sequence=target.value;await loadSamples(true);return}
    if(target.matches("[data-target-frame]")){await seekTargetFrame(Number(target.value));return}
    if(target.matches("[data-scrubber]")){await loadSample(Number(target.value));return}
    if(target.matches("[data-compare-record]")){if(target.checked){if(state.selected.size>=8){target.checked=false;return}state.selected.add(target.dataset.compareRecord)}else state.selected.delete(target.dataset.compareRecord);await refreshComparison();return}
  }
  function onInput(event){if(event.target.matches("[data-scrubber]")){if(state.playing)stopPlayback();const label=root.querySelector(".results-sample-count");if(label)label.textContent=`${Number(event.target.value)+1} / ${state.sampleTotal}`}if(event.target.matches("[data-avatar-opacity]")){state.avatarOpacity=Number(event.target.value);if(state.avatarContext){applyAvatarOpacity(state.avatarContext);state.scene?.render()}}}
  async function onClick(event){
    if(event.target.closest("[data-ui-help]"))return;
    if(await predictionDownload.handleClick(event))return;
    const button=event.target.closest("[data-action]");if(!button)return;const action=button.dataset.action;
    if(action==="selection-drawer"){setSelectionDrawerOpen(true);return}
    if(action==="close-selection-drawer"){setSelectionDrawerOpen(false);return}
    if(action==='open-prediction'){await loadRecord();return}
    if(action==="delete-test"){openDeleteDialog();return}
    if(action==="source"){await switchSource(button.dataset.resultSource);return}
    if(action==="refresh-results"){await loadModels();if(state.view==="compare"){await loadCompareRecords();render()}return}
    if(action==="compare"){setSelectionDrawerOpen(false);state.request++;state.sampleRequest++;state.predictionLoadingId=null;disposeVisual();state.view="compare";state.selected=new Set(state.record?[state.record.record_id]:[]);state.compareResult=null;onViewChange("compare");render();await loadCompareRecords();render();return}
    if(action==="back"){setSelectionDrawerOpen(false);state.view="results";onViewChange("results");render();return}
    if(action==="detail"){showDetail(button.dataset.detail);return}
    if(action==="reset-view"){state.scene?.reset();return}
    if(action==="first-sample"){await loadSample(0);return}
    if(action==="last-sample"){await loadSample(state.sampleTotal-1);return}
    if(action==="prev-sample"){await loadSample(Math.max(0,state.sampleIndex-1));return}
    if(action==="next-sample"){await loadSample(Math.min(state.sampleTotal-1,state.sampleIndex+1));return}
    if(action==="play-samples"){if(state.playing)stopPlayback();else await startPlayback();return}
    if(action==="sync-view"){const id=root.querySelector("[data-sync-sample]")?.value;if(id)await openSynchronizedSample(id);return}
  }
  async function openSynchronizedSample(sampleId){const ids=Array.from(state.selected);const results=[];for(const recordId of ids){try{results.push(await resultsApi("/api/results/sample?"+new URLSearchParams({record_id:recordId,sample_id:sampleId})))}catch(_error){results.push(null)}}const dialog=document.createElement("dialog");dialog.className="results-sync-dialog";const title=create("header","results-sync-heading",sampleId);title.append(actionButton("Close","close-sync","button button-quiet"));dialog.append(title);const grid=create("div","results-sync-grid");for(let i=0;i<results.length;i++){const item=create("section","results-sync-item");item.append(create("strong","",state.compareResult.records[i]?.test_id||ids[i]));const canvas=create("div","results-sync-canvas");item.append(canvas);grid.append(item);if(results[i]&&Array.isArray(results[i].radar)){try{const viewer=new SpatialView(canvas);viewer.setFrame({radar:list(results[i].radar),joints:list(results[i].target),edges:BONES,index:0,generation:0},{});viewer.fit();item._viewer=viewer}catch(_error){canvas.textContent="Sample unavailable"}}}dialog.append(grid);document.body.append(dialog);dialog.addEventListener("click",e=>{if(e.target.closest('[data-action="close-sync"]'))dialog.close()});dialog.addEventListener("close",()=>{for(const item of grid.children)item._viewer?.dispose();dialog.remove()},{once:true});dialog.showModal()}
  async function selectTest(testId){
    const wanted=String(testId||"");if(state.source!=="user")await switchSource("user");else await loadModels();let match=null,matchedModel=null,matchedWeight=null;
    for(const model of state.models){
      try{
        const payload=await resultsApi("/api/results/models/"+encodeURIComponent(model.id)+"/weights");
        for(const weight of list(payload.weights)){
          const test=list(weight.tests).find(item=>item.record_id===wanted||item.test_id===wanted);
          if(test){match=test;matchedModel=model.id;matchedWeight=weight;break}
        }
      }catch(_error){}
      if(match)break;
    }
    if(!match&&wanted.includes(":")){
      try{const record=await resultsApi("/api/results/record?"+new URLSearchParams({record_id:wanted}));matchedModel=record.model;match=record}catch(_error){}
    }
    if(!match)return false;
    if(matchedModel){
      try{
        const payload=await resultsApi("/api/results/models/"+encodeURIComponent(matchedModel)+"/weights");
        state.weights=list(payload.weights);matchedWeight=state.weights.find(item=>item.id===match.weight_id)||matchedWeight;
      }catch(_error){}
    }
    state.view="results";onViewChange("results");state.model=matchedModel;state.weight=matchedWeight?.id||match.weight_id;state.record=null;state.compareResult=null;
    await loadTests();
    const selected=state.records.find(item=>item.record_id===wanted||item.test_id===wanted||item.record_id===match.record_id);
    if(selected){state.record=selected;await loadRecord();return true}
    if(match.record_id){state.record=match;await loadRecord();return true}
    return false;
  }
  root._resultsUnmount=()=>{disposeSelectionDrawer();mobileResultsQuery.removeEventListener("change",syncSelectionDrawer);disposeVisual();root.removeEventListener("change",onChange);root.removeEventListener("click",onClick);root.removeEventListener("input",onInput)};
  let activated=false;
  return {activate:()=>{const entering=!state.active||!activated,refresh=activated&&!state.active;activated=true;state.active=true;predictionDownload.setActive(true);root.hidden=false;if(entering&&mobileResultsQuery.matches){clearTimeout(selectionDrawerOpenTimer);selectionDrawerOpenTimer=setTimeout(()=>{selectionDrawerOpenTimer=0;if(state.active&&mobileResultsQuery.matches)setSelectionDrawerOpen(true)},350)}if(refresh){return loadModels().then(async()=>{if(state.view==="compare"){await loadCompareRecords();render()}})}state.diagram?.setActive(true);state.scene?.resize()},deactivate:()=>{setSelectionDrawerOpen(false);state.active=false;predictionDownload.setActive(false);root.hidden=true;state.diagram?.setActive(false);stopPlayback();state.sampleRequest++;},selectTest,unmount:()=>{state.disposed=true;predictionDownload.dispose();removeLoadingActions();state.deleteDialog?.close();root._resultsUnmount?.()}};
}

export function mountComparisonPanel(root,{api,onViewChange=()=>{}}={}){
  return mountResultsPanel(root,{api,onViewChange,view:"compare"});
}
