/** Canonical, read-only presentation of recorded training parameters. */
export const DATASET_NAME='mmWave-Densepose';
const own=(value,key)=>Object.prototype.hasOwnProperty.call(value||{},key);
const object=value=>value&&typeof value==='object'&&!Array.isArray(value)?value:{};
const hiddenNames=new Set(['original_name','original_training_run_id','original_test_run_id','source_run_id','source_name','run_name','old_run_name','source_model','source_task','provenance','source_metadata','source_repository','source_checkpoint','source_train_relative','source_test_relative','layout_migration']);
const aliases={
 num_points:['radar_point_size','max_points','num_in_points'],
 input_dim:['sample_dim'],num_joints:['num_keypoints','num_key_points'],
 epochs:['effective_max_epochs','max_epochs'],optimizer:['optimizer_name'],
 seq_len:['t_max'],model:['resolved_model_name'],split_strategy:['data_split_strategy'],
};
const numeric=new Set(['num_points','input_dim','num_joints','epochs','seq_len','seed','batch_size','learning_rate','weight_decay','num_workers','torch_threads','num_devices','num_out_points','pcc_label_radius','split_chunk_size','min_valid_points','max_timestamp_gap_ms','scheduler_eta_min','scheduler_step_size','scheduler_gamma']);
const singleModels=new Set(['DGCNN','PointTransformer','PointMLP','PoseGraphNet','PointNet2Baseline']);
const nullValue=value=>value===null||value===undefined||typeof value==='string'&&/^(?:\s*|none|null|undefined|nan|—)$/i.test(value.trim());
function clean(value,key=''){
 if(key==='scheduler_name'&&typeof value==='string'&&value.trim().toLowerCase()==='none')return 'none';
 if(nullValue(value))return null;
 if(Array.isArray(value))return value.map(item=>clean(item));
 if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).filter(([name])=>!hiddenNames.has(name)).map(([name,item])=>[name,clean(item,name)]));
 if(typeof value==='number')return Number.isFinite(value)?value:null;
 if(typeof value==='string'){
  const text=value.trim();
  if(numeric.has(key)&&text!==''&&Number.isFinite(Number(text)))return Number(text);
  if((text.startsWith('[')&&text.endsWith(']'))||(text.startsWith('{')&&text.endsWith('}'))){try{return clean(JSON.parse(text),key)}catch{}}
 }
 return value;
}
const recorded=value=>value!==null&&value!==undefined&&value!==''&&(!Array.isArray(value)||value.length>0);
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function canonicalModel(value){
 return ({PartTokenFormer:'PASTNet',PartSkeletonSpatialNet:'PASTNet',PointNet2:'PointNet2Baseline',PointNet:'PointNet2Baseline',mmPoseFK:'MMPoseFK',PCCN:'GlobalLocalNetwork',PCC:'PCCCompletion'})[value]||value;
}
function dataRoot(value,official){
 return official&&typeof value==='string'?'data/mmwave_densepose':value;
}
function canonicalParameters(input,{official=false,model=null}={}){
 const values=clean(object(input)),conflicts={};
 // feature_dim was used as an input alias in old logs, but also as a hidden width.
 const dimensions=values.input_dim??values.sample_dim;
 const featureIsInput=own(values,'feature_dim')&&[3,4,5].includes(Number(values.feature_dim))&&
  (dimensions===null||dimensions===undefined||Number(dimensions)===Number(values.feature_dim));
 const mapping={...aliases,input_dim:[...aliases.input_dim,...(featureIsInput?['feature_dim']:[])]};
 for(const [key,names] of Object.entries(mapping)){
  const entries=[key,...names].filter(name=>own(values,name)).map(name=>[name,key==='model'?canonicalModel(values[name]):clean(values[name],key)]);
  const present=entries.filter(([,value])=>recorded(value));
  if(entries.length)values[key]=(present[0]||entries[0])[1];
  if(present.length>1&&present.some(([,value])=>!same(value,present[0][1])))conflicts[key]=Object.fromEntries(present);
  for(const name of names)delete values[name];
 }
 if(recorded(model))values.model=canonicalModel(model);
 else if(recorded(values.model))values.model=canonicalModel(values.model);
 if(own(values,'data_root'))values.data_root=dataRoot(values.data_root,official);
 values.dataset_name=DATASET_NAME;
 if(own(values,'task')){
  if(['keypoint','keypoint_muti','single_frame','temporal'].includes(values.task))values.task='pose';
  if(values.task==='dense')values.task='completion';
 }
 if(own(values,'device'))values.device=String(values.device).toLowerCase();
 else if(recorded(values.accelerator))values.device=String(values.accelerator).toLowerCase()==='gpu'?'cuda':String(values.accelerator).toLowerCase();
 delete values.accelerator;
 if(recorded(values.device)&&values.device.startsWith('gpu'))values.device=values.device.replace(/^gpu/,'cuda');
 return {values,conflicts};
}
const groupDefinitions=[
 ['run','Run'],['input','Input parameters'],['model','Model parameters'],
 ['training','Training parameters'],['runtime','Runtime parameters'],
 ['additional','Additional parameters'],['artifacts','Artifacts'],['paper','Official reference'],
];
function groupFor(key){
 if(/^(dataset_name|data_root|split_strategy|split_manifest|partition|split_chunk_size|max_timestamp_gap_ms|min_valid_points|seed|num_points|input_dim|seq_len|feature_stats|train_window_lengths|target_mode|many_to_many)$/.test(key))return 'input';
 if(/^(model|task|data_family|ablation|variant|num_joints|num_out_points|pcc_|dropout|num_part|num_memory|part_|cross_|temporal_|spatial_|relative_temporal|hidden|feature_dim|voxel_|num_classes|stacks)/.test(key))return 'model';
 if(/^(epochs|batch_size|learning_rate|weight_decay|optimizer|scheduler_|early_stopping|.*_loss_weight|precision|resume|initialization)/.test(key))return 'training';
 if(/^(device|num_devices|device_count|num_workers|torch_threads|cache_size|cuda|ddp|debug|fast_dev|log_|output|checkpoint|load$|pin_memory|save$|visual|test_only|limit_batches|allow_partial)/.test(key))return 'runtime';
 return 'additional';
}
function active(key,values){
 const model=values.model,completion=model==='PCCCompletion',temporal=!singleModels.has(model)&&!completion;
 if(['seq_len','max_timestamp_gap_ms','target_mode','many_to_many','train_window_lengths'].includes(key)&&!temporal)return false;
 if(key==='split_chunk_size'&&values.split_strategy==='subject'&&!recorded(values[key]))return false;
 if(key==='num_joints'&&completion)return false;
 if(key==='num_out_points'&&!['PCCCompletion','GlobalLocalNetwork'].includes(model))return false;
 if(key==='pcc_label_radius'&&!completion)return false;
 if(key==='ablation'&&model!=='PASTNet')return false;
 if(key==='torch_threads'&&!String(values.device||'').startsWith('cpu'))return false;
 if(key==='scheduler_eta_min'&&values.scheduler_name!=='cosine')return false;
 if(['scheduler_step_size','scheduler_gamma'].includes(key)&&values.scheduler_name!=='step')return false;
 return true;
}
export function buildRunParameterGroups(run={}){
 const actual={...object(run.hyperparameters),...object(run.hparams),...object(run.config)};
 const {values,conflicts}=canonicalParameters(actual,{official:run.origin==='official',model:run.model});
 const groups=Object.fromEntries(groupDefinitions.map(([id,title])=>[id,{id,title,primary:{},all:{}}]));
 const identity={run_id:run.training_run_id||run.run_id||run.id,time:run.time||run.timestamp,
  status:run.status,origin:run.origin==='official'?'Official training':'My training',action:run.action||run.task,
  asset_id:run.asset_id,weight_id:run.weight_id};
 for(const [key,value] of Object.entries(identity))if(value!==undefined)groups.run.all[key]=clean(value,key);
 for(const key of ['run_id','time','status','origin'])if(recorded(groups.run.all[key]))groups.run.primary[key]=groups.run.all[key];
 // Run/log association belongs to Run/Artifacts, not duplicated parameter cards.
 for(const key of ['run_id','training_run_id','weight_id','test_id','action','checkpoint','dataset_manifest_sha256','training_cache_identity','partition_cache_id','split_index_sha256'])delete values[key];
 for(const [key,value] of Object.entries(values))groups[groupFor(key)].all[key]=value;
 const important={
  input:['dataset_name','data_root','split_strategy','seed','partition','split_chunk_size','num_points','input_dim','seq_len'],
  model:['model','ablation','num_joints','num_out_points','pcc_label_radius'],
  training:['epochs','batch_size','learning_rate','weight_decay','optimizer','scheduler_name','scheduler_eta_min','scheduler_step_size','scheduler_gamma'],
  runtime:['device','num_workers','torch_threads'],
 };
 for(const [id,keys] of Object.entries(important))for(const key of keys){
  if(recorded(values[key])&&active(key,values)&&!own(conflicts,key))groups[id].primary[key]=values[key];
 }
 for(const [key,value] of Object.entries(conflicts)){
  const all=groups[groupFor(key)].all;
  (all.parameter_conflicts??={})[key]=value;
 }
 groups.artifacts.all=clean({...object(run.files),...(recorded(run.checkpoint)?{checkpoint:run.checkpoint}:{}),
  ...(recorded(run.config_file)?{config_file:run.config_file,config_id:run.config_id,config_sha256:run.config_sha256}:{})});
 if(recorded(run.config_file))groups.artifacts.primary.config_file=run.config_file;
 const reference=object(run.paper_configuration);
 if(Object.keys(reference).length)groups.paper.all=canonicalParameters(reference,{official:run.origin==='official',model:run.model}).values;
 return groupDefinitions.map(([id])=>groups[id]).filter(group=>Object.keys(group.all).length>0);
}
const labels={run_id:'Run ID',time:'Time',status:'Status',origin:'Origin',model:'Model',task:'Task family',data_family:'Data family',
 data_root:'Data root',dataset_name:'Dataset',split_strategy:'Split strategy',split_manifest:'Recorded split index',partition:'Train / validation / test ratio',
 split_chunk_size:'Chunk size (frames)',max_timestamp_gap_ms:'Maximum timestamp gap (ms)',min_valid_points:'Minimum valid points',
 num_points:'Input points',input_dim:'Input features',seq_len:'Window length',num_joints:'Output joints',num_out_points:'Completed points',
 pcc_label_radius:'PCC label radius',seed:'Seed',epochs:'Epochs',batch_size:'Batch size',learning_rate:'Learning rate',
 weight_decay:'Weight decay',optimizer:'Optimizer',ablation:'PAST-Net variant',scheduler_name:'Scheduler',
 scheduler_eta_min:'Minimum learning rate',scheduler_step_size:'Step size',scheduler_gamma:'Gamma',
 device:'Device',num_workers:'Data loader workers',torch_threads:'CPU threads',parameter_conflicts:'Recorded parameter conflicts',
 asset_id:'Asset ID',weight_id:'Weight ID',checkpoint:'Checkpoint'};
export function parameterLabel(key){return labels[key]||String(key).replaceAll('_',' ').replace(/\b\w/g,char=>char.toUpperCase())}
export function parameterValue(value,key=''){
 if(value===null||value===undefined||value==='')return key==='split_chunk_size'?'Whole sequence':'Not recorded';
 if(key==='split_strategy')return ({sequence_chunk:'Sequence-chunk',subject:'Subject-independent',intra_sequence:'Within-sequence (historical)',sequence:'Whole-sequence (historical)'})[value]||String(value);
 if(key==='data_family')return ({single_frame:'Single-frame pose',temporal:'Temporal pose',completion:'Point-cloud completion'})[value]||String(value);
 if(key==='device')return String(value).toUpperCase();
 if(key==='partition'&&Array.isArray(value))return value.join(' / ');
 if(key==='optimizer'&&String(value).toLowerCase()==='adam')return 'Adam';
 if(key==='scheduler_name')return ({cosine:'Cosine',step:'Step',none:'None'})[value]||String(value);
 if(typeof value==='object')return JSON.stringify(value,null,2);
 return String(value);
}
