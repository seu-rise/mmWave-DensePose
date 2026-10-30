import {api} from "./api.js";
import {datasetNeedsDownload,datasetBlocksModelTasks,datasetStatusText} from './dataset-status.js';
import {installContextHelp,createHelpIcon,createScienceDbLink} from "./ui-help.js";
import {withLoading,bindLoadingActions} from "./loading.js";
import {SpatialView} from "./scene.js";
import {setupTabs} from "./tabs.js";
import {showTimestamp} from "./time.js";
import {FramePrefetch} from "./frame-prefetch.js";
import {mountTrainingPanel} from "./training.js";
import {mountLogsPanel} from "./logs.js";
import {mountResultsPanel} from "./results.js";
import {mountPaperPanel} from "./paper.js";

const $ = id => document.getElementById(id);
document.documentElement.dataset.webguiMode = "pending";
const webguiCapabilities = api("/api/capabilities").then(capabilities => {
  if (!["local", "demo"].includes(capabilities.mode)) throw new Error("WebGUI permissions unavailable");
  document.documentElement.dataset.webguiMode = capabilities.mode;
  return capabilities;
}).catch(error => reportError(error, true));
installContextHelp();
function groupHelp(element,text,label,external=false){
  if(!element)return;
  const group=document.createElement('div');group.className='ui-entry-group';
  element.replaceWith(group);group.append(element);
  if(external)group.append(createScienceDbLink('Open dataset files on ScienceDB'));
  group.append(createHelpIcon(text,label));
}
groupHelp($('dataset-verify'),'Download, verify or repair the published dataset on this server.','Dataset installation help',true);
groupHelp($('dataset-overview'),'View dataset size, subjects, motions and sequence statistics.','Dataset overview help');
groupHelp($('spatial-title'),'Rotate the scene to inspect radar points and the recorded skeleton.','Point cloud help');
groupHelp(document.querySelector('.image-panel .panel-heading h2'),'View synchronized anonymized color, depth, infrared and body-index frames.','Synchronized images help');
document.querySelector('.timeline-top')?.append(createHelpIcon('Replay synchronized frames at a fixed 6.5 fps.','Playback help'));

function decorateDatasetDialog(dialog){
  const heading=dialog.querySelector('#dataset-dialog-title');
  if(!heading||heading.parentElement.classList.contains('ui-title-group'))return;
  const title=heading.textContent.toLowerCase();
  const text=/verif|scan/.test(title)?'Check installed files against the published release sizes and checksums.'
    :/overview/.test(title)?'View the dataset summary and its recorded sequence statistics.'
    :/repair/.test(title)?'Restore missing or changed files using only the affected dataset packages.'
    :'Download, install and verify the dataset on this server.';
  groupHelp(heading,text,'Dataset panel help',true);
  heading.parentElement.classList.add('ui-title-group');
}
const initialView=new URLSearchParams(location.search).get("view");
const PLAYBACK_FPS = 6.5;
const state = {files: [], summary: null, generation: null, index: 0, playing: false,
  loading: false, opening: false, frame: null, request: 0, epoch: 0, playbackEpoch: 0, active: true, stopped: false};
let view, frameController, playbackTimer, pollTimer;
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const badge = (label, tone = "") => `<span class="badge ${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
const imageURLs = new Map();
const prefetch = new FramePrefetch(prepareFrame);
let frameBytes = 12 * 1024 ** 2;
const transportIDs = ["first-frame", "previous-frame", "play", "next-frame", "last-frame", "frame-index", "frame-slider", "fit-view"];

function reportError(error, connection = false) {
  if (error.name === "AbortError") return;
  $("error-text").textContent = error.message || String(error);
  $("error-box").hidden = false;
  if (connection) {
    $("connection-label").textContent = "Disconnected";
    $("connection-light").classList.remove("connected");
    pause();
  }
}

function connected() {
  $("connection-label").textContent = document.documentElement.dataset.webguiMode === "demo" ? "Official demo" : "Local service";
  $("connection-light").classList.add("connected");
}

function updateControls() {
  const busy = state.loading || state.opening;
  const available = !!state.summary && !busy;
  for (const id of transportIDs) $(id).disabled = !available;
  const selected = $("sequence-select").value;
  $("sequence-select").disabled = busy || !state.files.length;
  const index = state.files.findIndex(file => file.id === selected);
  $("previous-file").disabled = busy || index <= 0;
  $("next-file").disabled = busy || index < 0 || index >= state.files.length - 1;
  if (available) {
    $("first-frame").disabled = $("previous-frame").disabled = state.index === 0;
    $("last-frame").disabled = $("next-frame").disabled = state.index >= state.summary.frame_count - 1;
  }
}

function clearImages(message = "No image loaded") {
  for (const tile of document.querySelectorAll("[data-modality]")) {
    const image = tile.querySelector("img");
    image.hidden = true; image.removeAttribute("src");
    const previous = imageURLs.get(image);
    if (previous) URL.revokeObjectURL(previous);
    imageURLs.delete(image);
    tile.querySelector(".image-placeholder").hidden = false;
    tile.querySelector(".image-placeholder").textContent = message;
    tile.querySelector(".image-timestamp").textContent = "—";
    tile.querySelector(".image-timestamp").removeAttribute("title");
  }
}

function layoutImages() {
  const observations = $("image-observations");
  const style = getComputedStyle(observations);
  const bounds = observations.getBoundingClientRect();
  const width = bounds.width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = bounds.height - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  if (width <= 0 || height <= 0) return;
  const ratio = (key, fallback) => Number(document.querySelector(`[data-modality="${key}"]`).dataset.imageRatio) || fallback;
  const color = document.querySelector('[data-modality="color_frames"]');
  const aux = document.querySelector(".aux-images");
  const auxKeys = ["depth_frames", "infrared_frames", "body_index_frames"];
  const auxRatios = auxKeys.map(key => ratio(key, 512 / 424));
  const columnGap = parseFloat(getComputedStyle(aux).columnGap) || 0;
  const rowGap = parseFloat(style.rowGap) || 0;
  const surface = getComputedStyle(color.querySelector(".image-surface"));
  const border = parseFloat(surface.borderLeftWidth) + parseFloat(surface.borderRightWidth);
  const auxWidthFitHeight = Math.max(0, width - 2 * columnGap - 3 * border)
    / auxRatios.reduce((sum, value) => sum + value, 0);
  const columns = auxRatios.map(value => `minmax(0, ${value * auxWidthFitHeight + border}fr)`).join(" ");
  observations.style.setProperty("--aux-columns", columns);
  const caption = color.querySelector("figcaption").getBoundingClientRect().height;
  const auxCaption = Math.max(...auxKeys.map(key => document.querySelector(`[data-modality="${key}"] figcaption`).getBoundingClientRect().height));
  const colorWidthFitHeight = Math.max(0, width - border) / ratio("color_frames", 16 / 9);
  const usableHeight = Math.max(0, height - caption - auxCaption - rowGap - 2 * border);
  // Maximize all four uncropped images at the existing pane width. If both
  // natural rows fit, RGB reaches full width and all auxiliary images reach
  // their shared full-width-row height. Short panes scale both rows equally.
  const scale = Math.min(1, usableHeight / Math.max(1, colorWidthFitHeight + auxWidthFitHeight));
  const colorHeight = caption + border + colorWidthFitHeight * scale;
  observations.style.setProperty("--color-row-height", `${colorHeight}px`);
}

let imageLayoutRequest;
function queueImageLayout() {
  cancelAnimationFrame(imageLayoutRequest);
  imageLayoutRequest = requestAnimationFrame(layoutImages);
}
const imageLayoutObserver = new ResizeObserver(queueImageLayout);
imageLayoutObserver.observe($("image-observations"));

function resetDataView() {
  state.request++;
  frameController?.abort();
  prefetch.clear();
  state.frame = null; state.index = 0;
  delete $("panel-dataset").dataset.sequenceId;
  view?.clear(); clearImages();
  $("scene-empty").hidden = false;
  $("subject-id").textContent = "—";
  $("frame-index").value = $("frame-slider").value = "0";
  $("frame-count").textContent = "/ —";
  $("point-count").textContent = "— points";
  $("joint-count").textContent = "— joints";
  $("radar-time").textContent = "Radar —";
  $("kinect-time").textContent = "Kinect —";
  $("radar-time").removeAttribute("title");
  $("kinect-time").removeAttribute("title");
  $("color-legend").hidden = true;
}

async function refreshFiles() {
  const listing = await api("/api/files");
  connected();
  const selected = $("sequence-select").value;
  state.files = listing.files;
  $("sequence-select").replaceChildren(new Option(listing.files.length ? "Choose a server sequence…" : "No supported sequence files found", ""));
  for (const file of listing.files) {
    const mb = (file.bytes / 1024 ** 2).toFixed(1);
    $("sequence-select").add(new Option(`${file.name}  ·  ${mb} MiB`, file.id));
  }
  if (state.files.some(file => file.id === selected)) $("sequence-select").value = selected;
  $("sequence-select").title = `${listing.root_label} / ${listing.files.length.toLocaleString()} sequence files`;
  updateControls();
}

async function openSequence() {
  const id = $("sequence-select").value;
  if (!id || state.loading || state.opening) return;
  state.opening = true;
  updateControls();
  pause();
  state.epoch++;
  $("error-box").hidden = true;
  try {
    await api("/api/open", {method: "POST", body: {id}});
    state.loading = true; state.summary = null; state.generation = null;
    resetDataView();
    $("loading-box").hidden = false;
    $("loading-message").textContent = "Checking and loading trusted sequence…";
    $("load-progress").value = 0;
    $("frame-message").textContent = "Loading original observations…";
    updateControls();
  } catch (error) { reportError(error); }
  finally { state.opening = false; updateControls(); }
}

let releaseSummary=null,verificationJob=null,verificationTimer=null;
let datasetInstallation=null,installationJob=null,installationTimer=null,datasetPending=false,datasetStatusError='',installationError='',datasetModalVersion=0,repairPlan=null,repairPlanError='',repairPlanLoading=false,datasetTransferAction='download',datasetRepairId=null,datasetStatusCheckedAt=0,datasetStatusRequest=null;
function bytesLabel(value){let n=Number(value);if(!Number.isFinite(n)||n<0)return '—';const u=['B','KiB','MiB','GiB','TiB'];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return `${n.toFixed(i?1:0)} ${u[i]}`}
function modalButton(label,action,cls=''){return `<button type="button" data-dataset-action="${escapeHtml(action)}" class="${cls}">${escapeHtml(label)}</button>`}
let datasetOverviewSplit='';
function datasetSplitMarkup(){
 const splits=releaseSummary?.release?.splits||[],selected=splits.find(item=>item.id===datasetOverviewSplit)||splits[0];
 if(!selected)return '<div class="pad muted">—</div>';
 datasetOverviewSplit=selected.id;const counts=selected.counts||{},total=['train','val','test'].reduce((n,k)=>n+Number(counts[k]||0),0);
 return `<div class="pad"><div class="row between"><span>Protocol</span><select data-release-split aria-label="Release split">${splits.map(item=>`<option value="${escapeHtml(item.id)}" ${item.id===selected.id?'selected':''}>${escapeHtml(item.id)}</option>`).join('')}</select></div><div class="dataset-split-bars">${['train','val','test'].map(key=>`<span data-split-part="${key}" data-percent="${total?100*Number(counts[key]||0)/total:0}"></span>`).join('')}</div><div class="row between dataset-split-counts">${['train','val','test'].map(key=>`<small>${key[0].toUpperCase()+key.slice(1)} ${Number(counts[key]||0).toLocaleString('en-US')}</small>`).join('')}</div><div class="dataset-modal-line"></div><dl class="dataset-meta-list"><dt>Window</dt><dd>${Number(selected.seq_len)} frames</dd><dt>Target frame</dt><dd>end − 1</dd></dl></div>`;
}
function layoutDatasetSplit(){for(const part of $('dataset-dialog').querySelectorAll('[data-split-part]'))part.style.width=part.dataset.percent+'%'}
function datasetSummaryModal(error=''){
 const dialog=$('dataset-dialog'),r=releaseSummary?.release||{},target=releaseSummary?.target_root_label||'—',last=verificationJob||releaseSummary?.latest;
 const paperFrameSplit=(r.splits||[]).find(item=>item.id==='single_intra_s42'&&item.seq_len===1),paperFrameCounts=paperFrameSplit?.counts||{},splitFrameTotal=['train','val','test'].every(key=>Number.isInteger(paperFrameCounts[key])&&paperFrameCounts[key]>=0)?['train','val','test'].reduce((sum,key)=>sum+paperFrameCounts[key],0):null;
 const validFrames=Number.isInteger(r.valid_synchronized_frames)?r.valid_synchronized_frames:(r.statistics_source==null?splitFrameTotal:null),validFramesSource=r.valid_frames_source||(r.statistics_source==null&&splitFrameTotal!==null?paperFrameSplit.id:null);
 const cards=[['Sequences',r.sequences],['Stored frames',r.stored_frames??r.frames],['Valid synchronized frames',validFrames],['Subjects',r.subjects],['Release size',bytesLabel(r.sequence_bytes)]],modalities=[['radar_frames','Radar'],['keypoint_frames','GT · 19 joints'],['color_frames','RGB'],['depth_frames','Depth'],['infrared_frames','Infrared'],['body_index_frames','Body index']];
 const statisticsNote=validFramesSource?`Counts use the frozen release baseline, not a live scan of this directory. The official-valid total is the sum of the frozen ${validFramesSource} train/validation/test one-frame samples; the official config requires strictly more than 50 finite-XYZ radar points per frame. Temporal-window splits also apply the timestamp-gap rule. Dataset verification separately checks local files against the baseline.`:'Counts use the frozen release baseline, not a live scan of this directory. The matching one-frame split is unavailable, so the official-valid total cannot be shown. Dataset verification separately checks local files against the baseline.';
 dialog.innerHTML=`<header class="dataset-dialog-head"><div><div class="eyebrow">RELEASE / ${escapeHtml(r.release_version||'—')}</div><h2 id="dataset-dialog-title">Dataset overview</h2></div><button type="button" class="close" data-dataset-action="close" aria-label="Close overview">×</button></header><div class="dataset-dialog-body">${error?`<div class="dataset-modal-error" role="alert">${escapeHtml(error)}</div>`:''}<div class="dataset-summary-grid">${cards.map(([label,value])=>`<section class="dataset-summary-card"><span>${label}</span><strong>${value==null?'—':escapeHtml(typeof value==='number'?value.toLocaleString('en-US'):value)}</strong></section>`).join('')}</div><p class="dataset-statistics-note">Counts are from the frozen release baseline, not a live scan of this directory. The valid-frame count is the sum of the frozen ${escapeHtml(r.valid_frames_source||'single_intra_s42')} train/validation/test split (one frame per sample), matching Official Table 2. Use dataset verification to check local files against this baseline.</p><div class="dataset-overview-meta"><section class="panel"><div class="panel-head"><h2>Modalities</h2></div><div class="pad dataset-modalities">${modalities.map(([key,label])=>`<div><span class="${(r.modalities||[]).includes(key)?'dataset-pass':'muted'}">●</span> ${label}</div>`).join('')}</div></section><section class="panel"><div class="panel-head"><h2>Split and windows</h2></div><div data-release-split-content>${datasetSplitMarkup()}</div></section></div><section class="panel"><div class="panel-head"><h2>Release and verification</h2></div><div class="pad"><dl class="dataset-meta-list"><dt>Directory</dt><dd>${escapeHtml(target)}</dd><dt>Release</dt><dd>${escapeHtml(r.release_version||'—')}</dd><dt>Verified</dt><dd>${last?badge(String(last.status).replaceAll('_',' '),last.status==='consistent'?'green':'orange'):'Not verified'}</dd></dl></div></section></div><footer class="dataset-dialog-foot"><div class="row">${error?modalButton('Retry','overview','primary'):''}${last&&last.status!=='scanning'?modalButton('View report','report'):''}${modalButton('Close','close')}${modalButton(datasetPrimaryLabel(),'manage','primary')}</div></footer>`;
 dialog.querySelector('.dataset-statistics-note').textContent=statisticsNote;layoutDatasetSplit();decorateDatasetDialog(dialog);if(!dialog.open)dialog.showModal();
}
function verificationProgressModal(){const dialog=$("dataset-dialog"),j=verificationJob||{},progress=j.total?Math.min(100,j.checked*100/j.total):null,message=`${j.stage||'Starting'} · ${Number(j.checked||0).toLocaleString()}${j.total?` / ${Number(j.total).toLocaleString()}`:''} · ${bytesLabel(j.bytes_hashed||0)}`;
 dialog.innerHTML=`<header class="dataset-dialog-head"><div><div class="eyebrow">DATASET / VERIFICATION</div><h2 id="dataset-dialog-title">Verifying dataset</h2></div><button class="close" data-dataset-action="close" aria-label="Close verification">×</button></header><div class="dataset-dialog-body"><div class="verification-progress-label"><span>${escapeHtml(message)}</span><span>${Number(j.errors||0)} errors · ${Number(j.warnings||0)} warnings</span></div><progress class="verification-progress" ${progress===null?'':`max="100" value="${progress}"`}></progress></div><footer class="dataset-dialog-foot"><button data-dataset-action="close">Close</button>${modalButton('Cancel scan','cancel','primary')}</footer>`;
 decorateDatasetDialog(dialog);if(!dialog.open)dialog.showModal()}
function displayDatasetValue(value){if(value===null||value===undefined)return '—';return typeof value==='object'?JSON.stringify(value):String(value)}
function datasetPrimaryLabel(){
 if(datasetInstallation?.active_job?.status==='running')return datasetInstallation.active_job.action==='repair'?'Repair progress':'Download progress';
 if(verificationJob?.status==='scanning')return 'Resume scan';
 if(datasetStatusError)return 'Retry dataset status';
 return datasetNeedsDownload(datasetInstallation)?datasetInstallation.availability==='partial'||datasetInstallation.can_resume?'Continue download':'Download dataset':datasetInstallation?'Verify dataset':'Dataset';
}
function updateDatasetPrimary(){const button=$('dataset-verify');button.textContent=datasetPrimaryLabel();button.disabled=datasetPending;button.setAttribute('aria-busy',String(datasetPending));const status=$('dataset-installation-status');if(status){status.textContent=datasetStatusText(datasetInstallation);status.hidden=!status.textContent;status.classList.toggle('is-incomplete',datasetBlocksModelTasks(datasetInstallation));}}
function stopDatasetPolling(){clearTimeout(verificationTimer);clearTimeout(installationTimer);datasetModalVersion++;}
async function refreshDatasetInstallation(){
 try{const previous=datasetInstallation?.installed_sequences;datasetInstallation=await api('/api/dataset/installation');datasetStatusCheckedAt=Date.now();datasetStatusError='';updateDatasetPrimary();if(previous!==undefined&&previous!==datasetInstallation.installed_sequences)await refreshFiles();return datasetInstallation}
 catch(error){datasetStatusError=error.message||String(error);updateDatasetPrimary();throw error}
}
function installationStage(stage){return ({queued:'Preparing',preparing:'Preparing',checking_cache:'Checking cached packages',verifying_cache:'Checking cached packages',downloading:'Downloading',downloaded:'Downloaded',extracting:'Extracting',installing:'Installing',verifying:'Verifying',complete:'Complete',cancelled:'Cancelled',interrupted:'Interrupted',failed:'Failed'})[stage]||String(stage||'Preparing').replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());}
function installationProgressModal(){
 const dialog=$('dataset-dialog'),job=installationJob,transferAction=job?.action||datasetTransferAction,progress=job?.progress||{},running=job?.status==='running',status=job?.status||'unavailable',stage=job?.stage||status,error=installationError||job?.error||(!job?datasetInstallation?.reason:'');
 const downloaded=Number(job?.bytes_downloaded||0),total=Number(job?.total_bytes||0),fileDownloaded=Number(progress.bytes_downloaded||0),fileTotal=Number(progress.bytes_total||0),checked=Number(progress.files_checked||0),filesTotal=Number(progress.files_total||0),completed=Number(job?.completed_archives||0),packages=Number(job?.total_archives||0);
 const steps=[progress.checking_cache?['checking_cache','Check cache']:['downloading','Download'],['extracting','Extract'],['installing','Install'],['verifying','Verify'],['complete','Complete']],stageIndex=progress.checking_cache?0:stage==='downloaded'?1:Math.max(0,steps.findIndex(([name])=>name===stage));
 const title=!job?(transferAction==='repair'?'Repair unavailable':'Download unavailable'):job.status==='complete'?(job.action==='repair'?'Dataset repaired':'Dataset ready'):job.status==='failed'?(job.action==='repair'?'Repair failed':'Download failed'):job.status==='cancelled'?'Operation cancelled':job.status==='interrupted'?'Operation interrupted':job.action==='repair'?'Repairing dataset':'Downloading dataset';
 const percent=total>0?Math.min(100,downloaded*100/total):null;
 dialog.innerHTML=`<header class="dataset-dialog-head"><div><div class="eyebrow">DATASET / ${transferAction==='repair'?'REPAIR':'DOWNLOAD'}</div><h2 id="dataset-dialog-title">${title}</h2></div><button type="button" class="close" data-dataset-action="close" aria-label="Close download">×</button></header><div class="dataset-dialog-body">${error?`<div class="dataset-modal-error" role="alert">${escapeHtml(typeof error==='string'?error:displayDatasetValue(error))}</div>`:''}<p class="dataset-transfer-readiness" role="status">${escapeHtml(datasetStatusText(datasetInstallation))}</p><div class="dataset-transfer-stages" aria-label="Installation stages">${steps.map(([name,label],index)=>`<div class="dataset-transfer-stage ${job?.status==='complete'||running&&index<stageIndex?'is-complete':running&&index===stageIndex?'is-current':''}"><span>${String(index+1).padStart(2,'0')}</span>${label}</div>`).join('')}</div><div class="dataset-transfer-status"><span class="verification-status ${status==='complete'?'consistent':status==='running'?'':status}">${escapeHtml(installationStage(job?.status==='running'?(progress.checking_cache?'checking_cache':stage):status))}</span><span>${packages?`${completed} / ${packages} packages`:''}</span></div><section class="dataset-transfer-card"><div class="verification-progress-label"><span>Downloaded</span><strong>${bytesLabel(downloaded)}${total?` / ${bytesLabel(total)}`:''}</strong></div><progress class="verification-progress" aria-label="Dataset download progress" ${percent===null?(running?'':'max="100" value="0"'):`max="100" value="${percent}"`}></progress><dl class="dataset-transfer-meta"><dt>Package</dt><dd class="mono">${escapeHtml(progress.filename||job?.filename||'—')}</dd><dt>Stage</dt><dd>${escapeHtml(installationStage(progress.checking_cache?'checking_cache':stage))}</dd>${running&&stage==='downloading'?`<dt>Current package</dt><dd>${bytesLabel(fileDownloaded)}${fileTotal?` / ${bytesLabel(fileTotal)}`:''}</dd>`:''}${filesTotal?`<dt>Files checked</dt><dd>${checked.toLocaleString('en-US')} / ${filesTotal.toLocaleString('en-US')}</dd>`:''}${job?.finished_at?`<dt>Finished</dt><dd>${new Date(job.finished_at*1000).toLocaleString()}</dd>`:''}</dl>${running&&stage==='downloading'&&fileTotal?`<progress class="verification-progress dataset-package-progress" aria-label="Current package download progress" max="${fileTotal}" value="${Math.min(fileTotal,fileDownloaded)}"></progress>`:''}</section></div><footer class="dataset-dialog-foot">${modalButton('Close','close')}<div class="row">${job?.verification_id?modalButton('View report','installation-report'):''}${running?`${installationError?modalButton('Retry progress','installation-resume'):''}<button type="button" data-dataset-action="installation-cancel" ${datasetPending?'disabled':''}>Cancel</button>`:modalButton(job?.status==='complete'?'Overview':job?(transferAction==='repair'?'Retry repair':'Continue download'):'Retry availability',job?.status==='complete'?'overview':transferAction==='repair'?'repair':'manage','primary')}</div></footer>`;
 decorateDatasetDialog(dialog);if(!dialog.open)dialog.showModal();
}
function repairPlanMarkup(){
 if(repairPlanLoading)return '<section class="dataset-repair-panel" role="status">Loading repair options…</section>';
 if(repairPlanError)return `<section class="dataset-repair-panel"><div class="dataset-modal-error" role="alert">${escapeHtml(repairPlanError)}</div>${modalButton('Retry repair options','repair-options')}</section>`;
 if(!repairPlan)return '';
 const packages=repairPlan.archives||[],manual=repairPlan.unrepairable||[];
 return `<section class="dataset-repair-panel"><div class="row between"><h3>Repair packages</h3><span class="badge">${packages.length} packages · ${bytesLabel(packages.reduce((total,item)=>total+Number(item.size_bytes||0),0))}</span></div>${repairPlan.reason?`<div class="dataset-repair-reason">${escapeHtml(repairPlan.reason)}</div>`:''}${packages.length?`<div class="dataset-repair-packages">${packages.map(item=>`<div class="dataset-repair-package"><div><strong class="mono">${escapeHtml(item.filename||item.id)}</strong><span>${bytesLabel(item.size_bytes)} · ${(item.affected_files||[]).length} affected files${item.available?'':' · Unavailable'}</span></div><details><summary>Files</summary><ul>${(item.affected_files||[]).map(path=>`<li class="mono">${escapeHtml(path)}</li>`).join('')}</ul></details></div>`).join('')}</div>`:''}${manual.length?`<div class="dataset-repair-manual"><strong>Manual review</strong><ul>${manual.map(item=>`<li>${escapeHtml(typeof item==='string'?item:[item.path,item.reason||item.resolution].filter(Boolean).join(' · '))}</li>`).join('')}</ul></div>`:''}</section>`;
}
function verificationReportModal(){const dialog=$('dataset-dialog'),j=verificationJob||{},findings=j.findings||[];
 const rows=findings.map(item=>`<tr><td class="mono">${escapeHtml(item.path)}</td><td>${escapeHtml(displayDatasetValue(item.expected))}<br><strong class="${item.severity==='error'?'dataset-error':item.severity==='warning'?'dataset-warning':'dataset-unverified'}">${escapeHtml(displayDatasetValue(item.actual))}</strong></td><td><span class="badge ${item.severity==='error'?'orange':''}">${escapeHtml(item.severity)}</span><br>${escapeHtml(item.cause)}</td><td>${escapeHtml(item.resolution)}</td></tr>`).join('');
 dialog.innerHTML=`<header class="dataset-dialog-head"><div><div class="eyebrow">${escapeHtml(j.release_version||'RELEASE')}</div><h2 id="dataset-dialog-title">Verification report</h2></div><button class="close" data-dataset-action="close" aria-label="Close report">×</button></header><div class="dataset-dialog-body"><div class="verification-counts"><span class="verification-status ${j.status==='consistent'?'consistent':'mismatch'}">${escapeHtml((j.status||'unavailable').replaceAll('_',' '))}</span><span class="badge">${Number(j.checked||0)} / ${Number(j.total||0)} checked</span><span class="badge orange">${Number(j.errors||0)} errors</span><span class="badge orange">${Number(j.warnings||0)} warnings</span><span class="badge">${Number(j.unverified||0)} unverified</span></div><dl class="dataset-report-meta"><dt>Target</dt><dd>${escapeHtml(j.target_root_label||releaseSummary?.target_root_label||'—')}</dd><dt>Started</dt><dd>${j.started_at?new Date(j.started_at*1000).toLocaleString():'—'}</dd><dt>Finished</dt><dd>${j.finished_at?new Date(j.finished_at*1000).toLocaleString():'—'}</dd></dl>${repairPlanMarkup()}<div class="table-wrap dataset-findings-scroll"><table class="verification-report-table"><thead><tr><th>Path</th><th>Expected / Actual</th><th>Severity / Cause</th><th>Resolution</th></tr></thead><tbody>${rows||'<tr><td colspan="4">No differences</td></tr>'}</tbody></table></div></div><footer class="dataset-dialog-foot"><button data-dataset-action="overview">Overview</button><div class="row">${(repairPlan?.archives||[]).length?`<button type="button" class="primary" data-dataset-action="repair" ${!repairPlan.can_repair||datasetPending?'disabled':''}>Repair ${repairPlan.archives.length} package${repairPlan.archives.length===1?'':'s'}</button>`:''}${modalButton('Verify again','verify','primary')}</div></footer>`;
 decorateDatasetDialog(dialog);if(!dialog.open)dialog.showModal();
}
async function loadDatasetRepairPlan(version=datasetModalVersion){
 if(!verificationJob?.id||!['mismatch','cannot_confirm','failed'].includes(verificationJob.status))return;
 const id=verificationJob.id;repairPlan=null;repairPlanError='';repairPlanLoading=true;verificationReportModal();
 try{const result=await api(`/api/dataset/repairs?verification_id=${encodeURIComponent(id)}`);if(version!==datasetModalVersion||verificationJob?.id!==id)return;repairPlan=result;}
 catch(error){if(version!==datasetModalVersion)return;repairPlanError=error.message||String(error)}
 finally{if(version===datasetModalVersion){repairPlanLoading=false;if($('dataset-dialog').open)verificationReportModal()}}
}
function showDatasetVerificationReport(){repairPlan=null;repairPlanError='';repairPlanLoading=false;verificationReportModal();loadDatasetRepairPlan();}
async function showDatasetOverview(){stopDatasetPolling();const version=datasetModalVersion;try{const results=await Promise.all([api('/api/dataset/summary'),refreshDatasetInstallation()]);if(version!==datasetModalVersion)return;releaseSummary=results[0];verificationJob=releaseSummary.latest;datasetSummaryModal()}catch(error){if(version===datasetModalVersion)datasetSummaryModal(error.message)}}
async function handleDatasetPrimaryAction(){
 if(datasetPending)return;const version=datasetModalVersion;datasetPending=true;updateDatasetPrimary();
 try{await refreshDatasetInstallation();if(version!==datasetModalVersion)return;if(datasetInstallation.active_job?.status==='running'){installationJob=datasetInstallation.active_job;datasetTransferAction=installationJob.action;datasetRepairId=installationJob.verification_id||null;installationError='';stopDatasetPolling();installationProgressModal();pollDatasetInstallation();return}if(datasetInstallation.availability!=='missing'){const summary=await api('/api/dataset/summary');if(version!==datasetModalVersion)return;releaseSummary=summary;verificationJob=releaseSummary.latest}if(verificationJob?.status==='scanning'){resumeDatasetVerification();return}if(datasetNeedsDownload(datasetInstallation)){if(!datasetInstallation.can_download){stopDatasetPolling();installationJob=null;datasetTransferAction='download';installationError=datasetInstallation.reason||'Download links are not published.';installationProgressModal();return}await startDatasetInstallation('download',true)}else await startDatasetVerification(true)}
 catch(error){if(version===datasetModalVersion){installationJob=null;installationError=error.message||String(error);installationProgressModal()}}
 finally{datasetPending=false;updateDatasetPrimary()}
}
async function startDatasetInstallation(action,alreadyPending=false){
 if(datasetPending&&!alreadyPending)return;datasetPending=true;updateDatasetPrimary();stopDatasetPolling();const version=datasetModalVersion;installationError='';datasetTransferAction=action;
 try{const body={action};if(action==='repair'){const id=verificationJob?.id||installationJob?.verification_id||datasetRepairId;if(!id)throw Error('Verify the dataset before repairing packages.');body.verification_id=id;datasetRepairId=id}const job=await api('/api/dataset/installations',{method:'POST',body});installationJob=job;datasetInstallation={...datasetInstallation,active_job:job};updateDatasetPrimary();if(version!==datasetModalVersion)return;installationProgressModal();pollDatasetInstallation(version)}
 catch(error){if(version===datasetModalVersion){installationJob=null;installationError=error.message||String(error);installationProgressModal()}}
 finally{datasetPending=false;updateDatasetPrimary()}
}
async function pollDatasetInstallation(version=datasetModalVersion){
 clearTimeout(installationTimer);if(!$('dataset-dialog').open||version!==datasetModalVersion||!installationJob?.id)return;
 try{const job=await api(`/api/dataset/installations/${encodeURIComponent(installationJob.id)}`);if(version!==datasetModalVersion||!$('dataset-dialog').open)return;installationJob=job;installationError='';datasetInstallation={...datasetInstallation,active_job:job.status==='running'?job:null,latest_job:job};installationProgressModal();updateDatasetPrimary();if(job.status==='running')installationTimer=setTimeout(()=>pollDatasetInstallation(version),500);else{await refreshDatasetInstallation();if(version===datasetModalVersion&&$('dataset-dialog').open)installationProgressModal();if(job.status==='complete')refreshFiles().catch(error=>reportError(error))}}
 catch(error){if(version!==datasetModalVersion||!$('dataset-dialog').open)return;installationError=error.message||String(error);installationProgressModal()}
}
async function cancelDatasetInstallation(){
 if(datasetPending||!installationJob?.id)return;datasetPending=true;installationProgressModal();updateDatasetPrimary();const version=datasetModalVersion;
 try{installationJob=await api(`/api/dataset/installations/${encodeURIComponent(installationJob.id)}/cancel`,{method:'POST',body:{}});if(version===datasetModalVersion&&$('dataset-dialog').open){installationProgressModal();pollDatasetInstallation(version)}}
 catch(error){if(version===datasetModalVersion){installationError=error.message||String(error);installationProgressModal()}}
 finally{datasetPending=false;updateDatasetPrimary();if(version===datasetModalVersion&&$('dataset-dialog').open)installationProgressModal()}
}
async function showInstallationVerification(){
 if(!installationJob?.verification_id)return;stopDatasetPolling();const version=datasetModalVersion;
 try{const report=await api(`/api/dataset/verifications/${encodeURIComponent(installationJob.verification_id)}`);if(version!==datasetModalVersion)return;verificationJob=report;showDatasetVerificationReport()}
 catch(error){if(version===datasetModalVersion){installationError=error.message||String(error);installationProgressModal()}}
}
async function startDatasetVerification(alreadyPending=false){
 if(datasetPending&&!alreadyPending)return;datasetPending=true;updateDatasetPrimary();stopDatasetPolling();const version=datasetModalVersion;repairPlan=null;repairPlanError='';
 try{const job=await api('/api/dataset/verifications',{method:'POST',body:{}});verificationJob=job;updateDatasetPrimary();if(version!==datasetModalVersion)return;verificationProgressModal();pollDatasetVerification(version)}
 catch(error){if(version===datasetModalVersion)datasetSummaryModal(error.message)}
 finally{datasetPending=false;updateDatasetPrimary()}
}
function resumeDatasetVerification(){stopDatasetPolling();verificationProgressModal();pollDatasetVerification();}
async function pollDatasetVerification(version=datasetModalVersion){
 clearTimeout(verificationTimer);if(!$('dataset-dialog').open||version!==datasetModalVersion)return;
 try{const job=await api(`/api/dataset/verifications/${encodeURIComponent(verificationJob.id)}`);if(version!==datasetModalVersion||!$('dataset-dialog').open)return;verificationJob=job;updateDatasetPrimary();if(job.status==='scanning'){verificationProgressModal();verificationTimer=setTimeout(()=>pollDatasetVerification(version),500)}else{showDatasetVerificationReport();refreshDatasetInstallation().catch(()=>{})}}
 catch(error){if(version===datasetModalVersion&&$('dataset-dialog').open)datasetSummaryModal(error.message)}
}
async function cancelDatasetVerification(){
 if(datasetPending)return;datasetPending=true;updateDatasetPrimary();const version=datasetModalVersion;
 try{const job=await api(`/api/dataset/verifications/${encodeURIComponent(verificationJob.id)}/cancel`,{method:'POST',body:{}});if(version!==datasetModalVersion)return;verificationJob=job;verificationProgressModal();verificationTimer=setTimeout(()=>pollDatasetVerification(version),450)}catch(error){if(version===datasetModalVersion)datasetSummaryModal(error.message)}finally{datasetPending=false;updateDatasetPrimary()}
}
async function handleDatasetModalAction(event){
 const el=event.target.closest('[data-dataset-action]');if(!el||el.disabled)return;const action=el.dataset.datasetAction;
 if(action==='close')$('dataset-dialog').close();else if(action==='manage')await handleDatasetPrimaryAction();else if(action==='verify')await startDatasetVerification();else if(action==='resume')resumeDatasetVerification();else if(action==='cancel')await cancelDatasetVerification();else if(action==='report')showDatasetVerificationReport();else if(action==='overview')await showDatasetOverview();else if(action==='repair')await startDatasetInstallation('repair');else if(action==='repair-options')await loadDatasetRepairPlan();else if(action==='installation-resume'){installationError='';installationProgressModal();pollDatasetInstallation()}else if(action==='installation-cancel')await cancelDatasetInstallation();else if(action==='installation-report')await showInstallationVerification();
}

function updateLegend() {
  const legend = view?.legend;
  const visible = !!state.frame && legend?.available && $("show-radar").checked;
  $("color-legend").hidden = !visible;
  const doppler = $("color-mode").value === "doppler";
  $("doppler-min").disabled = $("doppler-max").disabled = !doppler;
  if (!visible) return;
  $("legend-min").textContent = Number(legend.low.toFixed(2));
  $("legend-max").textContent = Number(legend.high.toFixed(2));
  $("legend-gradient").style.background = `linear-gradient(90deg,${legend.stops.join(",")})`;
  $("color-note").textContent = doppler ? "Doppler · stored values" : "SNR · fixed range";
}

function renderOptions() {
  $("point-size-value").textContent = `${Number($("point-size").value).toFixed(1)}×`;
  const low = Number($("doppler-min").value), high = Number($("doppler-max").value);
  if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
    $("doppler-max").setCustomValidity("Maximum must be greater than minimum");
    $("doppler-max").reportValidity(); return;
  }
  $("doppler-max").setCustomValidity("");
  view?.setOptions({color: $("color-mode").value, radar: $("show-radar").checked,
    skeleton: $("show-skeleton").checked, boundary: $("show-bounds").checked,
    pointSize: Number($("point-size").value), dopplerMin: low, dopplerMax: high});
  updateLegend();
}

async function prepareImage(tile, frame, signal) {
  const key = tile.dataset.modality;
  const prepared = {tile, url: null, image: null,
    message: "No image for this frame"};
  if (!frame.images[key]) return prepared;

  // Keep the currently displayed frame untouched while fetching and decoding.
  // Each request owns at most one off-DOM replacement per image channel.
  const release = () => {
    signal.removeEventListener("abort", release);
    if (prepared.url) URL.revokeObjectURL(prepared.url);
    prepared.image?.removeAttribute("src");
    prepared.url = null; prepared.image = null;
  };
  prepared.release = release;
  prepared.transfer = () => {
    signal.removeEventListener("abort", release);
    prepared.url = null; prepared.image = null;
  };
  try {
    const blob = await api(frame.images[key].url, {signal, image: true});
    if (signal.aborted) throw new DOMException("Frame request cancelled", "AbortError");
    prepared.url = URL.createObjectURL(blob);
    prepared.image = new Image();
    signal.addEventListener("abort", release, {once: true});
    prepared.image.src = prepared.url;
    await prepared.image.decode();
    prepared.bytes = blob.size + prepared.image.naturalWidth * prepared.image.naturalHeight * 4;
    if (signal.aborted) throw new DOMException("Frame request cancelled", "AbortError");
  } catch (error) {
    release();
    if (signal.aborted || error.name === "AbortError") throw new DOMException("Frame request cancelled", "AbortError");
    prepared.message = "Unreadable image data";
  }
  return prepared;
}

async function prepareFrame(index, generation, signal) {
  let images = [];
  try {
    const frame = await api(`/api/frame?generation=${generation}&index=${index}`, {signal});
    images = await Promise.all([...document.querySelectorAll("[data-modality]")]
      .map(tile => prepareImage(tile, frame, signal)));
    if (signal.aborted) throw new DOMException("Frame request cancelled", "AbortError");
    return {frame, images, bytes: images.reduce((sum, image) => sum + (image.bytes || 0), 0)
      + frame.radar.length * 64 + (frame.joints?.length || 0) * 24,
      release() { for (const image of images) image.release?.(); }};
  } catch (error) {
    for (const image of images) image.release?.();
    throw error;
  }
}

function commitImage(prepared, frame) {
  const tile = prepared.tile, previousImage = tile.querySelector("img");
  const previousURL = imageURLs.get(previousImage);
  const placeholder = tile.querySelector(".image-placeholder");
  showTimestamp(tile.querySelector(".image-timestamp"), frame.image_timestamps[tile.dataset.modality]);
  if (prepared.image) {
    // Insert the actual decoded element. Changing src on the visible element
    // can briefly show an empty image even when another Image decoded that URL.
    const replacement = prepared.image;
    for (const attribute of previousImage.attributes) {
      if (attribute.name !== "src" && attribute.name !== "hidden") replacement.setAttribute(attribute.name, attribute.value);
    }
    replacement.hidden = false;
    const ratio = replacement.naturalWidth / replacement.naturalHeight;
    if (Number.isFinite(ratio) && ratio > 0 && tile.dataset.imageRatio !== String(ratio)) {
      tile.dataset.imageRatio = String(ratio);
      tile.style.setProperty("--image-ratio", String(ratio));
      queueImageLayout();
    }
    imageURLs.set(replacement, prepared.url);
    previousImage.replaceWith(replacement);
    prepared.transfer();
    placeholder.hidden = true;
  } else {
    previousImage.hidden = true;
    previousImage.removeAttribute("src");
    placeholder.textContent = prepared.message;
    placeholder.hidden = false;
  }
  imageURLs.delete(previousImage);
  if (previousURL) URL.revokeObjectURL(previousURL);
}

async function showFrame(index) {
  if (!state.summary || state.loading) return;
  index = Math.max(0, Math.min(Math.trunc(Number(index) || 0), state.summary.frame_count - 1));
  const request = ++state.request, generation = state.generation;
  frameController?.abort();
  const controller = new AbortController(); frameController = controller;
  let prepared;
  try {
    prepared = await prefetch.take(index, generation, controller.signal);
    if (request !== state.request || generation !== state.generation || controller.signal.aborted || state.loading || state.stopped) return;
    const {frame, images} = prepared;

    // One synchronous commit: scene, frame metadata and every image belong to
    // the same frame, with no loading-placeholder paint between valid frames.
    state.frame = frame; state.index = index;
    view?.setFrame(frame, state.summary);
    $("scene-empty").hidden = !!view;
    $("frame-index").value = $("frame-slider").value = String(index);
    $("point-count").textContent = `${frame.radar.length.toLocaleString()} points`;
    $("joint-count").textContent = `${(frame.joints || []).filter(point => point.every(Number.isFinite)).length} joints`;
    showTimestamp($("radar-time"), frame.radar_timestamp, "Radar ");
    showTimestamp($("kinect-time"), frame.kinect_timestamp, "Kinect ");
    for (const image of images) commitImage(image, frame);
    // setFrame already applies the current options and renders the scene.
    updateLegend(); updateControls();
    frameBytes = prepared.bytes;
    prefetch.advance(index, generation, frameBytes);
    const mode = $("color-mode").value;
    $("frame-message").textContent = mode !== "solid" && !view?.legend?.available
      ? `${mode === "doppler" ? "Doppler" : "SNR"} unavailable · Solid color` : "Original frame order · Zero-based index";
  } catch (error) {
    if (error.name === "AbortError") return;
    pause();
    if (error.status !== 409) reportError(error);
    $("frame-message").textContent = "Frame unavailable · Check sequence status";
  } finally {
    prepared?.release();
  }
}

function pause() {
  if (state.playing) { state.request++; frameController?.abort(); }
  state.playing = false; state.playbackEpoch++; clearTimeout(playbackTimer);
  prefetch.clear();
  $("play").textContent = "▶";
  $("play").setAttribute("aria-label", "Play");
  $("play").setAttribute("aria-pressed", "false");
}

async function tick(epoch) {
  if (epoch !== state.playbackEpoch || !state.playing || !state.active || state.loading || !state.summary) return;
  if (state.index >= state.summary.frame_count - 1) { pause(); return; }
  const start = performance.now();
  await showFrame(state.index + 1);
  if (state.playing && epoch === state.playbackEpoch) playbackTimer = setTimeout(() => tick(epoch), Math.max(0, 1000 / PLAYBACK_FPS - (performance.now() - start)));
}

async function togglePlayback() {
  if (state.playing) { pause(); return; }
  if (!state.summary || state.loading) return;
  state.playing = true;
  const epoch = ++state.playbackEpoch;
  $("play").textContent = "Ⅱ";
  $("play").setAttribute("aria-label", "Pause");
  $("play").setAttribute("aria-pressed", "true");
  if (state.index >= state.summary.frame_count - 1) await showFrame(0);
  if (!state.playing || epoch !== state.playbackEpoch || !state.active || state.loading) return;
  prefetch.begin(state.generation, state.index + 1, state.summary.frame_count - 1, frameBytes);
  await prefetch.warm();
  if (!state.playing || epoch !== state.playbackEpoch || !state.active || state.loading) return;
  playbackTimer = setTimeout(() => tick(epoch), 1000 / PLAYBACK_FPS);
}

async function poll() {
  const epoch = state.epoch;
  try {
    const status = await api("/api/status");
    if (epoch !== state.epoch) return;
    connected();
    const wasLoading = state.loading;
    state.loading = status.loading;
    $("loading-box").hidden = !status.loading;
    $("load-progress").value = status.progress;
    $("loading-message").textContent = status.message;
    if (status.loading) {
      pause();
    } else if (status.sequence && (status.generation !== state.generation || wasLoading)) {
      pause(); resetDataView();
      state.summary = status.sequence; state.generation = status.generation;
      $("panel-dataset").dataset.sequenceId = status.sequence.file_id;
      $("panel-dataset").dataset.generation = String(status.generation);
      $("subject-id").textContent = status.sequence.subject_id;
      $("sequence-select").value = status.sequence.file_id;
      const last = String(status.sequence.frame_count - 1);
      $("frame-index").max = $("frame-slider").max = last;
      $("frame-count").textContent = `/ ${last} (${status.sequence.frame_count.toLocaleString()} frames)`;
      view?.reset();
      await showFrame(0);
    } else if (!status.sequence && !status.loading && (wasLoading || status.generation !== state.generation)) {
      pause(); state.summary = null; state.generation = status.generation; resetDataView();
      $("panel-dataset").dataset.generation = String(status.generation);
      $("frame-message").textContent = status.message;
    }
    if(Date.now()-datasetStatusCheckedAt>3000&&!datasetStatusRequest&&!datasetPending){
      datasetStatusRequest=refreshDatasetInstallation().catch(()=>{}).finally(()=>{datasetStatusRequest=null});
    }
    if (status.error) reportError(new Error(status.error));
    updateControls();
  } catch (error) { reportError(error, true); }
  finally { if (!state.stopped) pollTimer = setTimeout(poll, state.loading ? 250 : 1200); }
}

try { view = new SpatialView($("scene")); }
catch (error) {
  $("scene-empty").querySelector("h3").textContent = "WebGL rendering unavailable";
  $("scene-empty").querySelector("p").textContent = "Enable WebGL2 in a current browser. Data and image inspection remain available.";
  reportError(error);
}

let workbenchMountPromise;
let initialNavigation = true;
let activeWorkbenchTab = "dataset";
let workbenchSections;
let workbenchControllers = {};
const workbenchPanels = ["training", "logs", "results", "paper"];
function updateWorkspaceURL(name){const url=new URL(location.href);url.searchParams.set("view",({paper:"official",training:"tasks"})[name]||name);history.replaceState(null,"",url)}
function showWorkbenchSection(name) {
  updateWorkspaceURL(name);
  if (!workbenchSections) return;
  state.active = false;
  $("panel-dataset").hidden = true;
  $("panel-workbench").hidden = false;
  const topTab=name==='logs'?'training':name;
  document.body.dataset.view=name;
  for(const tab of document.querySelectorAll('.workbench-header .tabs [role="tab"]')){
    const selected=tab.id==='tab-'+topTab;tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;
  }
  for (const key of workbenchPanels) workbenchSections[key].hidden = key !== name;
  let activeReady=Promise.resolve();
  for (const key of workbenchPanels) {
    const panel = workbenchControllers[key];
    if (panel?.setActive){const ready=panel.setActive(key === name);if(key===name&&ready?.then)withLoading(ready).catch(reportError);}
    else if (key === name){const ready=panel?.activate?.();if(ready?.then)activeReady=withLoading(ready).catch(reportError);}
    else panel?.deactivate?.();
  }
  activeWorkbenchTab = name;
  document.getElementById("panel-workbench").setAttribute("aria-labelledby",
    name === "training" || name === "logs" ? "tab-training" : name === "results" ? "tab-results" : "tab-paper");
  return activeReady;
}
function mountWorkbenchOnce() {
  if (!workbenchMountPromise) workbenchMountPromise=mountWorkbenchPanels();
  return workbenchMountPromise;
}
async function mountWorkbenchPanels() {
  const host = $("workbench-content");
  host.innerHTML = workbenchPanels.map(key => `<section id="tool-${key}" class="tool-page" ${key === "training" ? "" : "hidden"}><div id="${key}-root"></div></section>`).join("");
  workbenchSections = Object.fromEntries(workbenchPanels.map(key => [key, $(`tool-${key}`)]));
  const capabilities = await webguiCapabilities;
  const initialSource = capabilities?.mode === "demo" ? "official" : "user";
  const register=(name,factory)=>{
    const finish=controller=>{
      workbenchControllers[name]=controller;
      if(controller?.setActive)controller.setActive(activeWorkbenchTab===name);
      else if(activeWorkbenchTab===name)controller?.activate?.();else controller?.deactivate?.();
      return controller;
    };
    try{
      const controller=factory();
      return controller?.then?controller.then(finish).catch(error=>{
        $(`${name}-root`).replaceChildren(Object.assign(document.createElement('div'),{className:'connection-error',textContent:error.message||'Panel unavailable'}));
      }):Promise.resolve(finish(controller));
    }catch(error){
      $(`${name}-root`).textContent=error.message||'Panel unavailable';return Promise.resolve();
    }
  };
  await Promise.all([
      register('training',()=>mountTrainingPanel($("training-root"), {api, onTabChange: async (tab, testId) => {
        if (tab === "results") {
          await mountWorkbenchOnce();
          await showWorkbenchSection("results");
          if (testId) await workbenchControllers.results?.selectTest?.(testId);
          return;
        }
        const view = tab === "execution" ? "training" : "logs";
        showWorkbenchSection(view);
        if (view === "logs") document.getElementById("tab-training").setAttribute("aria-selected", "true");
      }})),
      register('logs',()=>mountLogsPanel($("logs-root"), {api,initialOrigin:initialSource,onTabChange:()=>workbenchControllers.training?.showExecution?.()})),
      register('results',()=>mountResultsPanel($("results-root"), {api,initialSource,view:initialView==='compare'?'compare':'results',onViewChange:view=>{showWorkbenchSection('results');if(view==='compare')updateWorkspaceURL('compare')}})),
      register('paper',()=>mountPaperPanel($("paper-root"), {api})),
    ]);
}
function showWorkspace(tab) {
  const requested=initialNavigation?initialView:null;
  updateWorkspaceURL(requested==='compare'?'compare':tab);
  initialNavigation=false;
  activeWorkbenchTab = tab;
  document.body.dataset.view=tab;
  const dataset = tab === "dataset";
  state.active = dataset;
  if (dataset) {
    pause();
    for(const controller of Object.values(workbenchControllers)){
      if(controller?.setActive)controller.setActive(false);else controller?.deactivate?.();
    }
    requestAnimationFrame(() => view?.resize());
    if(requested==='overview')showDatasetOverview();
    else refreshDatasetInstallation().catch(()=>{});
    return;
  }
  pause();
  withLoading(mountWorkbenchOnce()).catch(reportError);
  if (tab === "training"&&requested==='logs')showWorkbenchSection('logs');
  else if (tab === "training") workbenchControllers.training?.showExecution?.();
  else showWorkbenchSection(tab);
}
setupTabs(showWorkspace);
bindLoadingActions($("dataset-overview"),{click:showDatasetOverview});
bindLoadingActions($("dataset-verify"),{click:handleDatasetPrimaryAction});
bindLoadingActions($("dataset-dialog"),{click:handleDatasetModalAction});
$("dataset-dialog").addEventListener("change",event=>{if(event.target.matches("[data-release-split]")){datasetOverviewSplit=event.target.value;document.querySelector("[data-release-split-content]").innerHTML=datasetSplitMarkup();layoutDatasetSplit()}});
$("dataset-dialog").addEventListener("close",stopDatasetPolling);
bindLoadingActions($("sequence-select"),{change:async()=>{updateControls();await openSequence()}});
bindLoadingActions($("refresh-files"),{click:()=>refreshFiles().catch(error=>reportError(error,true))});
for (const [id, direction] of [["previous-file", -1], ["next-file", 1]]) {
  bindLoadingActions($(id),{click:async()=>{
    const index = state.files.findIndex(file => file.id === $("sequence-select").value);
    const file = state.files[index + direction];
    if (file) { $("sequence-select").value = file.id; updateControls(); await openSequence(); }
  }});
}
bindLoadingActions($("cancel-load"),{click:()=>api("/api/cancel",{method:"POST",body:{}}).catch(reportError)});
bindLoadingActions($("reconnect"),{click:async()=>{
  $("error-box").hidden = true;
  try { await refreshFiles(); } catch (error) { reportError(error, true); }
}});
for (const id of ["color-mode", "show-radar", "show-skeleton", "show-bounds", "point-size", "doppler-min", "doppler-max"]) {
  $(id).addEventListener("input", renderOptions);
}
$("reset-view").addEventListener("click", () => view?.reset());
$("fit-view").addEventListener("click", () => view?.fit());
bindLoadingActions($("play"),{click:togglePlayback});
for (const [id, index] of [["first-frame", () => 0], ["previous-frame", () => state.index - 1], ["next-frame", () => state.index + 1], ["last-frame", () => state.summary.frame_count - 1]]) {
  bindLoadingActions($(id),{click:async()=>{pause();await showFrame(index())}});
}
for (const [id, event] of [["frame-index", "change"], ["frame-slider", "input"]]) {
  if(event==='change')bindLoadingActions($(id),{change:async()=>{pause();await showFrame($(id).value)}});
  else $(id).addEventListener(event, () => { pause(); showFrame($(id).value); });
}
document.addEventListener("keydown", event => {
  if (activeWorkbenchTab !== "dataset" || !state.active || event.ctrlKey || event.metaKey || event.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if ([" ", "arrowleft", "arrowright", "r", "k", "b", "d"].includes(key)) event.preventDefault();
  if (key === " ") togglePlayback();
  if (key === "arrowleft" || key === "arrowright") { pause(); showFrame(state.index + (key === "arrowleft" ? -1 : 1)); }
  if (key === "r") view?.reset();
  for (const [letter, id] of [["k", "show-skeleton"], ["b", "show-bounds"]]) {
    if (key === letter) $(id)?.click();
  }
  if (key === "d") { $("color-mode").selectedIndex = ($("color-mode").selectedIndex + 1) % 3; renderOptions(); }
});
document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
window.addEventListener("pagehide", () => {
  state.stopped = true; pause(); clearTimeout(pollTimer); stopDatasetPolling(); frameController?.abort();
  prefetch.clear();
  cancelAnimationFrame(imageLayoutRequest); imageLayoutObserver.disconnect();
  clearImages(); view?.dispose();
});
// Pasting a restarted server's token URL may change only the fragment, not load
// a new document. Reinitialize transport and all state in that case.
window.addEventListener("hashchange", () => location.reload());
renderOptions();
refreshFiles().catch(error => reportError(error, true)).finally(poll);
