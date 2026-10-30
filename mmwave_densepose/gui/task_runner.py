"""Task catalog and process adapters for the authenticated local workbench."""
from __future__ import annotations
from collections import deque
import hashlib,json,math,os,re,signal,subprocess,sys,tempfile,threading,time,uuid
from pathlib import Path

from mmwave_densepose.workflows.catalog import COMPLETION_MODEL,ROOT,TASKS,action_for_model,get_catalog,list_registered_models,model_slug
from mmwave_densepose.engine.artifacts import artifact_roots, checkpoint_log_dir, resolve_run_paths
from mmwave_densepose.engine.devices import validate_gpu_selection
from mmwave_densepose.engine.run_config import describe_run_config, explicit_run_arguments, save_run_config
from mmwave_densepose.data.installation_state import DatasetIncompleteError, ensure_complete_dataset
from . import deleted_logs
from .validation_diagnostics import enrich_issue, exception_issue, safe_value

FIELDS={"model","task","data_root","split_strategy","split_chunk_size","partition","max_timestamp_gap_ms","min_valid_points","num_points","num_out_points","input_dim","num_joints","seq_len","seed","batch_size","epochs","learning_rate","weight_decay","optimizer","ablation","checkpoint","scheduler_name","scheduler_eta_min","scheduler_step_size","scheduler_gamma","completion_checkpoint","pcc_label_radius","dense_label_mode","num_workers","torch_threads","cache_size","device","gpu_ids"}
ACTIONS=set(TASKS)
ABSOLUTE_PATH=re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']+")
QUOTED_ABSOLUTE_PATH=re.compile(r"([\"'])[A-Za-z]:[\\/][^\"'\r\n]*\1")
MAX_REQUEST_ID=160

def _sha(value):return hashlib.sha256(value).hexdigest()
def _sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for block in iter(lambda:s.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def _atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def _redact(text):
 text=QUOTED_ABSOLUTE_PATH.sub(lambda match:match.group(1)+'<local-path>'+match.group(1),text.rstrip('\r\n'))
 return ABSOLUTE_PATH.sub('<local-path>',text)
class TaskRunner:
 """Serialized local CLI execution with resumable status and append-only logs."""
 def __init__(self,data_root,job_root=None,output_root=None,*,trust_pickle=False,max_parallel=1,project_root=None):
  self.project_root=Path(project_root or ROOT).resolve();self.data_root=Path(data_root).expanduser().resolve();self.artifact_roots=artifact_roots(self.project_root);self.output_root=self.project_root/'mmwave_densepose'/'logs';self.job_root=Path(job_root or self.artifact_roots['runtime']/'jobs').expanduser().resolve();self.trust_pickle=bool(trust_pickle);self.jobs_dir=self.job_root/'runs';self.jobs_dir.mkdir(parents=True,exist_ok=True)
  self._lock=threading.RLock();self._wake=threading.Condition(self._lock);self._queue=deque();self._jobs={};self._procs={};self._cancel={};self._requests={};self._validations={};self._reserved=set();self._closed=False;self._load_jobs();self._thread=threading.Thread(target=self._dispatch,name='workbench-task-queue',daemon=True);self._thread.start()
 def _dir(self,jid):return self.jobs_dir/jid
 def _job_dir(self,jid):return self._dir(jid)
 def _state_path(self,jid):return self._dir(jid)/'job.json'
 def _log_path(self,jid):return self._dir(jid)/'console.ndjson'
 def _load_jobs(self):
  for path in sorted(self.jobs_dir.glob('*/job.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    job=json.loads(path.read_text(encoding='utf-8'));jid=job.get('job_id')
    if not isinstance(jid,str) or not re.fullmatch(r'job_[A-Za-z0-9_-]{1,90}',jid):continue
    if job.get('status') in {'Queued','Starting','Running','Stopping'}:job.update(status='Needs review',phase='interrupted',error='Process ownership was lost when the service restarted.',updated_at=time.time());_atomic_json(path,job)
    self._jobs[jid]=job
    if job.get('request_id'):self._requests[job['request_id']]=jid
   except (OSError,ValueError,TypeError):continue
  self.prune_deleted_logs(force=True)
 def forget_deleted_outputs(self,paths,*,recovery_root=None,confirmed_dead_job_ids=()):
  """Retire inactive jobs after their exact log leaves were deleted."""
  outputs=set()
  for value in paths:
   output=deleted_logs.output_path(self.project_root,value)
   if output is None or output.exists():raise ValueError('Only deleted canonical log directories can be forgotten.')
   outputs.add(output)
  confirmed=set(confirmed_dead_job_ids)
  with self._lock:
   records=[]
   for jid,job in self._jobs.items():
    if deleted_logs.record_output(self.project_root,job) not in outputs:continue
    observed=deleted_logs.observed_record(self.project_root,job);process=self._procs.get(jid)
    verified_legacy=(jid in confirmed and job.get('status')=='Needs review' and not (observed or job).get('process_identity') and (process is None or process.poll() is not None))
    if verified_legacy or deleted_logs.inactive(job,observed=observed,process=process):records.append((jid,job))
   removed=deleted_logs.archive(self.project_root,'jobs',records,self._dir,recovery_root=recovery_root)
   for jid in removed:
    job=self._jobs.pop(jid);output=deleted_logs.record_output(self.project_root,job)
    self._cancel.pop(jid,None);self._procs.pop(jid,None);self._reserved.discard(output);self._validations.pop(job.get('report_id'),None)
   self._queue=deque(jid for jid in self._queue if jid not in removed)
   self._requests={request:jid for request,jid in self._requests.items() if jid not in removed}
   return {'deleted_job_ids':removed}
 def prune_deleted_logs(self,*,force=False):
  with self._lock:
   now=time.monotonic()
   if not force and now-getattr(self,'_retention_at',-math.inf)<1:return {'deleted_job_ids':[]}
   self._retention_at=now;receipts=deleted_logs.receipts(self.project_root);outputs=set()
   for jid,job in self._jobs.items():
    observed=deleted_logs.observed_record(self.project_root,job)
    if deleted_logs.inactive(job,observed=observed,process=self._procs.get(jid)) and deleted_logs.was_deleted(self.project_root,job,receipts,observed=observed):
     outputs.add(deleted_logs.record_output(self.project_root,job))
   try:return self.forget_deleted_outputs(outputs)
   except (OSError,ValueError):return {'deleted_job_ids':[],'retention_error':'Execution metadata could not be moved to recovery storage.'}
 @staticmethod
 def _parts(draft):
  if not isinstance(draft,dict):raise ValueError('Run parameters must be an object.')
  action=draft.get('action');config=dict(draft.get('config',{}));output=draft.get('output',draft.get('output_dir',config.pop('output_dir','')))
  if action not in ACTIONS:raise ValueError('Select a supported task.')
  if not isinstance(output,str):raise ValueError('Output directory must be relative.')
  options={k:draft.get(k,config.pop(k,None)) for k in ('initialization','limit_batches','allow_partial','save_predictions','visual','evaluation_weight_id')}
  return action,config,output,options
 def evaluation_weights(self,model):
  return self._evaluation_catalog().list_weights(model)
 def _evaluation_catalog(self):
  if not hasattr(self,'_evaluation_weight_catalog'):
   from mmwave_densepose.web.evaluation_weights import EvaluationWeights
   self._evaluation_weight_catalog=EvaluationWeights(self.project_root,self.data_root)
  return self._evaluation_weight_catalog
 def _artifact_paths(self,action,config,checkpoint,output):
  paths=resolve_run_paths(action,config,checkpoint=checkpoint,output=output or None,root=self.project_root)
  if output:return paths
  reserved=self._reserved_outputs()|self._reserved
  if paths['log_dir'] not in reserved:return paths
  if action=='train':
   match=re.search(r'_r([0-9]+)$',paths['run_id']);base=paths['run_id'][:match.start()] if match else paths['run_id'];number=int(match.group(1))+1 if match else 2
   parent=self.artifact_roots['train']/config['model'];prefix=base+'_r';digits=2
  else:
   match=re.fullmatch(r'(test|val|predict)_([0-9]+)',paths['log_dir'].name)
   if not match:raise ValueError('Invalid evaluation directory identity.')
   number=int(match.group(2))+1;parent=paths['log_dir'].parent;prefix=match.group(1)+'_';digits=3
  while True:
   candidate=parent/f'{prefix}{number:0{digits}d}';number+=1
   if candidate.exists() or candidate in reserved:continue
   try:return resolve_run_paths(action,config,checkpoint=checkpoint,output=candidate,root=self.project_root)
   except FileExistsError:continue
 def _layout_public(self,planned):
  checkpoint=planned.get('checkpoint_path')
  if checkpoint:
   checkpoint=Path(checkpoint);checkpoint=(checkpoint if checkpoint.is_absolute() else self.project_root/checkpoint).resolve()
  return {'origin':'user','run_id':planned['run_id'],'training_run_id':planned.get('training_run_id') or planned['run_id'],'test_id':planned.get('test_id'),
          'weight_id':planned.get('weight_id') or planned['run_id'],'weight_source':planned.get('weight_source') or ('official' if planned['run_id'].startswith('official_') else 'user'),
          'output_dir':Path(planned['log_dir']).relative_to(self.project_root).as_posix(),
          'checkpoint_path':checkpoint.relative_to(self.project_root).as_posix() if checkpoint else None}
 def preview_layout(self,draft):
  action,provided,output,options=self._parts(draft);model=provided.get('model')
  provided,_selection=self._evaluation_catalog().bind(action,provided,options)
  model=provided.get('model')
  if model not in list_registered_models():raise ValueError('Select a registered model.')
  rows=[item for item in get_catalog()['templates'] if item['model']==model]
  preference={'PASTNet':'main-pastnet','PointNet2Baseline':'ablation-pointnet','DGCNN':'main-dgcnn','PointTransformer':'main-point-transformer','PointMLP':'main-pointmlp','MMPoseFK':'main-mm-pose-fk','DualPathCNNBiLSTM':'main-dual-path','PntPoseAT':'main-pnt-pose-at','PoseGraphNet':'main-pose-graph','GlobalLocalNetwork':'main-global-local','PCCCompletion':'main-pcc-dense'}.get(model)
  template=next((item['config'] for item in rows if item['id']==preference),rows[0]['config'] if rows else {})
  config={**template,**{key:value for key,value in provided.items() if key in FIELDS},'model':model}
  if 'split_manifest' in provided:raise ValueError('Select split_strategy: sequence_chunk or subject; cache names are not split strategies.')
  from mmwave_densepose.data.partitions import normalize_data_config
  config=normalize_data_config(config)
  checkpoint=None if action=='train' and options.get('initialization','scratch')=='scratch' else config.get('checkpoint')
  return self._layout_public(self._artifact_paths(action,config,checkpoint,output))
 def _split_metadata(self,config,manifest):
  """Inspect the parameter-keyed cache registry without materializing data."""
  from mmwave_densepose.data.partitions import inspect_partition
  full=manifest.get('dataset_kind')=='full_multimodal'
  ensure_complete_dataset(self.data_root,manifest)
  selection=inspect_partition(self.data_root,config)
  pending=bool(selection.get('pending'))
  if pending and full and not self.trust_pickle:
   raise ValueError('Automatic data preparation requires operator trust for the installed dataset; restart the service with --trust-pickle.')
  metadata=dict(selection.get('metadata') or {})
  metadata['cache_id']=selection.get('cache_id')
  return metadata,pending
 def _validate(self,draft):
  errors=[];warnings=[];info=[]
  try:action,provided,output_value,options=self._parts(draft)
  except (ValueError,TypeError) as exc:return self._bad(exc)
  try:ensure_complete_dataset(self.data_root)
  except DatasetIncompleteError as exc:return self._bad(exc)
  try:provided,evaluation_weight=self._evaluation_catalog().bind(action,provided,options)
  except (ValueError,TypeError) as exc:
   issue=exception_issue(exc,name='checkpoint');return self._bad(exc,issue=issue)
  model=provided.get('model')
  for flag in ('allow_partial','save_predictions','visual'):
   if options.get(flag) is not None and not isinstance(options[flag],bool):errors.append(dict(name=flag,group='output' if flag in ('visual','save_predictions') else 'runtime',severity='error',message='Expected true or false.'))
  raw_limit=options.get('limit_batches')
  if raw_limit not in (None,'',0,'0'):
   try:
    if isinstance(raw_limit,bool) or int(raw_limit)<1 or float(raw_limit)!=int(raw_limit):raise ValueError()
    options['limit_batches']=int(raw_limit)
   except (TypeError,ValueError,OverflowError):errors.append(dict(name='limit_batches',group='runtime',severity='error',message='Batch limit must be a positive integer.'))
  if model not in list_registered_models():errors.append(dict(name='model',group='model',severity='error',message='Select a registered model.'));model='PASTNet'
  task=action_for_model(model,action);catalog=get_catalog();templates=[x for x in catalog['templates'] if x['model']==model]
  preference={'PASTNet':'main-pastnet','PointNet2Baseline':'ablation-pointnet','DGCNN':'main-dgcnn','PointTransformer':'main-point-transformer','PointMLP':'main-pointmlp','MMPoseFK':'main-mm-pose-fk','DualPathCNNBiLSTM':'main-dual-path','PntPoseAT':'main-pnt-pose-at','PoseGraphNet':'main-pose-graph','GlobalLocalNetwork':'main-global-local','PCCCompletion':'main-pcc-dense'}.get(model)
  template=next((x['config'] for x in templates if x['id']==preference),templates[0]['config'] if templates else {'model':model});unknown=set(provided)-FIELDS
  for key in sorted(unknown):errors.append(dict(name=key,group='model',severity='error',message='Unsupported configuration parameter.'))
  c={**template,**{k:v for k,v in provided.items() if k in FIELDS},'model':model,'task':task}
  c.setdefault('data_root',str(self.data_root));c.setdefault('num_points',80 if task=='completion' else 300);c.setdefault('num_out_points',196);c.setdefault('input_dim',3 if task=='completion' else 5);c.setdefault('num_joints',19);c.setdefault('seed',42);c.setdefault('batch_size',128);c.setdefault('epochs',100 if task=='completion' else 200);c.setdefault('learning_rate',.0005);c.setdefault('weight_decay',.00001);c.setdefault('optimizer','adam');c.setdefault('ablation','full');c.setdefault('scheduler_name','cosine');c.setdefault('scheduler_eta_min',.000001);c.setdefault('scheduler_step_size',30);c.setdefault('scheduler_gamma',.1);c.setdefault('num_workers',0);c.setdefault('torch_threads',4);c.setdefault('cache_size',2);c.setdefault('device','cpu')
  try:
   from mmwave_densepose.data.partitions import normalize_data_config
   c=normalize_data_config(c)
  except (ValueError,TypeError) as exc:
   errors.append(exception_issue(exc,name='split_strategy',group='data'))
  self._check_numbers(c,task,action,errors,warnings)
  public_root=str(c.get('data_root','configured-data-root'))
  try:
   root=Path(c['data_root']).expanduser();root=(root if root.is_absolute() else self.project_root/root).resolve()
   if root!=self.data_root or not root.is_dir():raise ValueError('Dataset root is outside this service data root.')
   public_root=root.relative_to(self.project_root).as_posix() if root.is_relative_to(self.project_root) else f'configured-data-root/{root.name}';c['data_root']=public_root;m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
   if not isinstance(m.get('sequences'),dict) or not m['sequences']:raise ValueError('The dataset manifest has no valid sequence inventory.')
  except (OSError,ValueError,TypeError,AssertionError) as exc:errors.append(dict(name='data_root',group='data',severity='error',message=str(exc)));m={}
  completion=task=='completion'
  meta=None;prepare_pending=False
  try:
   meta,prepare_pending=self._split_metadata(c,m)
   if meta.get('seq_len') is not None and meta['seq_len']!=int(c['seq_len']):errors.append(dict(name='seq_len',group='data',severity='error',message=f"Cache requires {meta.get('seq_len')} frames."))
   if meta.get('seed') is not None and meta['seed']!=int(c['seed']):errors.append(dict(name='seed',group='data',severity='error',message='Cache seed does not match the requested seed.'))
   if (m.get('partial') or meta.get('partial')) and not options.get('allow_partial'):errors.append(dict(name='allow_partial',group='data',severity='error',message='The selected dataset is partial.'))
   if prepare_pending:info.append(dict(name='cache',group='runtime',severity='info',message='The model, strategy, seed and data parameters select a new cache; it will be prepared automatically when the worker starts.'))
  except (OSError,ValueError,TypeError,KeyError) as exc:errors.append(exception_issue(exc,name='split_strategy',group='data'))
  split_name='train' if action=='train' else 'val' if action=='validate' else 'test'
  counts=meta.get('split_num_samples') if meta else None
  count=int(counts.get(split_name,0)) if isinstance(counts,dict) else None
  limit=options.get('limit_batches')
  try:
   if limit not in (None,'',0,'0') and int(limit)>0:
    if count is not None:count=min(count,int(limit)*int(c['batch_size']))
    warnings.append(dict(name='limit_batches',group='runtime',severity='warning',message='Limited run is incomplete.'))
  except (ValueError,TypeError):errors.append(dict(name='limit_batches',group='runtime',severity='error',message='Batch limit must be a positive integer.'))
  if count is not None and count<1:errors.append(dict(name='split_strategy',group='data',severity='error',message=f'{split_name.title()} split is empty.'))
  try:
   batch_size=max(1,int(c['batch_size']))
   if action=='train' and count is not None and count%batch_size==1:errors.append(dict(name='batch_size',group='training',severity='error',message='Final training batch has one sample; BatchNorm requires at least two.'))
  except (ValueError,TypeError,OverflowError):pass
  info.append(dict(name='split',group='data',severity='info',message=f'{count} {split_name} samples.' if count is not None else f'{split_name.title()} sample count will be determined during cache preparation.'))
  visual=bool(options.get('visual',False))
  if visual and (action!='evaluate' or completion):errors.append(dict(name='visual',group='output',severity='error',message='Visual PKL is available for pose Test only.'))
  ckpt,csha=self._resolve_checkpoint(c.get('checkpoint'),errors,'checkpoint','model',c) if not (action=='train' and options.get('initialization','scratch')=='scratch') else (None,None)
  if ckpt:c['checkpoint']=ckpt.relative_to(self.project_root).as_posix()
  if action=='train' and c.get('model')=='GlobalLocalNetwork':
   if not c.get('completion_checkpoint'):
    errors.append(dict(name='completion_checkpoint',group='model',severity='error',message='GlobalLocalNetwork training requires a PCC completion checkpoint.'))
   else:
    cc,ccsha=self._resolve_checkpoint(c.get('completion_checkpoint'),errors,'completion_checkpoint','model',dict(c,model='PCCCompletion',num_points=80,num_out_points=c.get('num_out_points',196),input_dim=3,seq_len=1))
    if cc:c['completion_checkpoint']=cc.relative_to(self.project_root).as_posix();c['_completion_sha256']=ccsha
  paths=None
  try:
   with self._lock:paths=self._artifact_paths(action,c,ckpt,output_value)
   out=paths['log_dir']
  except (OSError,ValueError,TypeError) as exc:
   errors.append(dict(name='output_dir',group='output',severity='error',message=str(exc)));out=self.output_root/'invalid'
  request_id=draft.get('request_id') if isinstance(draft.get('request_id'),str) else None
  with self._lock:
   if out in self._reserved_outputs() or out in self._reserved:errors.append(dict(name='output_dir',group='output',severity='error',message='Output directory is reserved by another task.'))
  if out.exists() and (not out.is_dir() or any(out.iterdir())):errors.append(dict(name='output_dir',group='output',severity='error',message='Output must be an empty directory; existing runs are never overwritten.'))
  try:config_runtime_cache=int(c.get('cache_size',2))
  except (ValueError,TypeError):errors.append(dict(name='cache_size',group='runtime',severity='error',message='Cache size must be a positive integer.'));config_runtime_cache=2
  if config_runtime_cache<1:errors.append(dict(name='cache_size',group='runtime',severity='error',message='Cache size must be a positive integer.'))


  if not __import__('importlib.util',fromlist=['find_spec']).find_spec('torch'):
   errors.append(dict(name='environment',group='runtime',severity='error',message='PyTorch is unavailable in the current environment.'))
  else:
   try:
    import torch
    if str(c.get('device','cpu')).startswith('cuda') and not torch.cuda.is_available():errors.append(dict(name='device',group='runtime',severity='error',message='CUDA is unavailable. Select CPU or install the CUDA profile.'))
   except (ImportError,RuntimeError):errors.append(dict(name='environment',group='runtime',severity='error',message='PyTorch could not be initialized.'))
  if options.get('allow_partial') and m.get('partial'):warnings.append(dict(name='allow_partial',group='runtime',severity='warning',message='Partial dataset is enabled.'))
  if action=='train':
   try:
    if float(c.get('learning_rate',0))>.01:warnings.append(dict(name='learning_rate',group='training',severity='warning',message='Learning rate exceeds 0.01.'))
   except (ValueError,TypeError):pass
   try:
    if int(c['epochs'])>1000:warnings.append(dict(name='epochs',group='training',severity='warning',message='Training exceeds 1000 epochs.'))
   except (KeyError,ValueError,TypeError):pass
  run_id=paths['run_id'] if paths else None
  weight_id=(paths.get('weight_id') or paths['run_id']) if paths else None
  test_id=paths['test_id'] if paths else None
  c.pop('_split_metadata',None);c.pop('_split_sample_count',None);c.pop('_checkpoint_sha256',None);c.pop('_completion_sha256',None)
  # Exclude fields hidden by the active model/task capability map.
  training_fields={'epochs','learning_rate','weight_decay','optimizer','ablation','scheduler_name','scheduler_eta_min','scheduler_step_size','scheduler_gamma'}
  model_fields={'num_out_points','pcc_label_radius','completion_checkpoint'}
  effective={k:v for k,v in c.items() if k not in ('visual','limit_batches','allow_partial','save_predictions','output_dir','initialization') and (action=='train' or k not in training_fields) and (k not in model_fields or k=='completion_checkpoint' and model=='GlobalLocalNetwork' or k=='num_out_points' and model in ('GlobalLocalNetwork','PCCCompletion') or k=='pcc_label_radius' and model=='PCCCompletion') and (k!='ablation' or model=='PASTNet') and (k not in ('num_joints','num_out_points','pcc_label_radius') or k=='num_joints' and model!='PCCCompletion' or k=='num_out_points' and model in ('GlobalLocalNetwork','PCCCompletion') or k=='pcc_label_radius' and model=='PCCCompletion') and (k not in ('torch_threads','cache_size') or k=='cache_size' or str(c.get('device','cpu'))=='cpu') and (k!='checkpoint' or not (action=='train' and options.get('initialization','scratch')=='scratch'))}
  effective['task']='completion' if completion else 'pose'
  signature={'action':action,'config':effective,'output':out.relative_to(self.project_root).as_posix(),'visual':visual,'save_predictions':bool(options.get('save_predictions') or action=='infer'),'limit_batches':int(limit) if limit not in (None,'') else None,'allow_partial':bool(options.get('allow_partial')),'checkpoint_sha256':csha,'cache_id':meta.get('cache_id') if meta else None}
  if evaluation_weight:signature.update(evaluation_weight_id=evaluation_weight['id'],evaluation_weight_signature=evaluation_weight['weight_signature'])
  draft_hash=_sha(json.dumps(signature,sort_keys=True,separators=(',',':'),allow_nan=False).encode()) if not errors else ''
  errors=[enrich_issue(issue,config=c,redact=_redact) for issue in errors];warnings=[enrich_issue(issue,config=c,redact=_redact) for issue in warnings]
  report={'report_id':'validation_'+uuid.uuid4().hex,'draft_hash':draft_hash,'valid':not errors and not warnings,'action':action,'model':c['model'],'config':effective,'effective_config':effective,'output_dir':out.relative_to(self.project_root).as_posix(),'output_path':str(out),'run_id':run_id,'weight_id':weight_id,'test_id':test_id,'visual':visual,'errors':errors,'warnings':warnings,'info':info,'signature':signature,'request_id':request_id,'checked_at':time.time(),'cache_id':meta.get('cache_id') if meta else None,'cache_pending':prepare_pending,'_config_internal':dict(c)}
  if evaluation_weight:report.update(evaluation_weight_id=evaluation_weight['id'],locked_fields=evaluation_weight['locked_fields'],editable_fields=evaluation_weight['editable_fields'])
  if paths:report.update(self._layout_public(paths))
  report['command_preview']=[]
  if not errors:
   try:
    report.update(describe_run_config(action,self._run_configuration(report),root=self.project_root,**self._run_options(report)))
    report['command_preview']=self._command_preview(report)
   except (OSError,ValueError,TypeError,RuntimeError,ImportError) as exc:
    errors.append(enrich_issue(exception_issue(exc),config=c,redact=_redact));report['valid']=False
  report['command_shell']='powershell' if os.name=='nt' else 'posix'
  with self._lock:
   if report['valid']:self._validations[report['report_id']]=report
   while len(self._validations)>200:self._validations.pop(next(iter(self._validations)))
  report['config']['data_root']=public_root;report['effective_config']=report['config'];report['_config_internal']['data_root']=str(self.data_root)
  public={k:v for k,v in report.items() if k not in ('output_path','signature','_config_internal')}
  return public if report['valid'] else safe_value(public,_redact)
 def _check_numbers(self,c,task,action,errors,warnings):
  def num(field,label,minimum=0,integer=False,group='runtime',strict=False):
   try:
    value=float(c[field]);valid=math.isfinite(value) and (value>minimum if strict else value>=minimum) and (not integer or value.is_integer())
    if not valid:raise ValueError()
    c[field]=int(value) if integer else value
   except (KeyError,TypeError,ValueError,OverflowError):
    constraint=f'{"an integer" if integer else "a finite number"} {"greater than" if strict else "greater than or equal to"} {minimum}'
    errors.append(dict(name=field,group=group,severity='error',message=f'{label} must be {constraint}.',expected=constraint,actual=c.get(field),suggestion=f'Set {label.lower()} to {constraint}. Restore selected weight settings if this parameter is locked.'))
  for f,l,g in [('num_points','Input points','data'),('seq_len','Window length','data'),('batch_size','Batch size','training'),('num_workers','Data loader workers','runtime'),('seed','Seed','data')]:num(f,l,0,True,g,f not in ('seed','num_workers'))
  if c.get('device') not in ('cpu','cuda') and not re.fullmatch(r'cuda:\d+',str(c.get('device',''))):
   errors.append(dict(name='device',group='runtime',severity='error',message='Select CPU or CUDA.'))
  try:
   if action=='train' and str(c.get('device','')).startswith('cuda') and 'gpu_ids' in c and not c['gpu_ids']:
    raise ValueError('Select at least one available GPU.')
   c['gpu_ids']=validate_gpu_selection(action,c.get('device','cpu'),c.get('gpu_ids',[]),c.get('batch_size'))
  except (ValueError,TypeError,RuntimeError) as exc:
   errors.append(dict(name='gpu_ids',group='runtime',severity='error',message=str(exc)))
  if action=='train' and c.get('optimizer','adam')!='adam':
   errors.append(dict(name='optimizer',group='training',severity='error',message='This release supports the Adam optimizer.'))
  if c.get('model')=='PCCCompletion':
   for f,l,g in [('num_out_points','Completed points','model')]:num(f,l,0,True,g,True)
   num('pcc_label_radius','PCC label radius',0,False,'data',True)
  elif c.get('model')=='GlobalLocalNetwork':num('num_out_points','Completed points',0,True,'model',True)
  if action=='train':
   if c.get('scheduler_name') not in ('cosine','step','none'):errors.append(dict(name='scheduler_name',group='schedule',severity='error',message='Use cosine, step or none.'))
   num('epochs','Epochs',0,True,'training',True);num('learning_rate','Learning rate',0,False,'training',True);num('weight_decay','Weight decay',0,False,'training')
   if c.get('model')=='PASTNet' and c.get('ablation','full') not in ('full','spatial_only','baseline'):errors.append(dict(name='ablation',group='model',severity='error',message='PAST-Net supports Full, Spatial only or baseline.'))
   if c.get('scheduler_name')=='cosine':num('scheduler_eta_min','Minimum learning rate',0,False,'schedule')
   if c.get('scheduler_name')=='step':
    num('scheduler_step_size','Step size',0,True,'schedule',True);num('scheduler_gamma','Gamma',0,False,'schedule',True)
    try:
     if float(c['scheduler_gamma'])>1:raise ValueError()
    except (KeyError,ValueError,TypeError):errors.append(dict(name='scheduler_gamma',group='schedule',severity='error',message='Gamma must be at most 1.'))
   try:
    if float(c['scheduler_eta_min'])>float(c['learning_rate']):errors.append(dict(name='scheduler_eta_min',group='schedule',severity='error',message='Minimum learning rate cannot exceed the initial learning rate.'))
   except (KeyError,ValueError,TypeError):pass
  if task=='completion':
   if not isinstance(c.get('input_dim'),int):
    try:c['input_dim']=int(c['input_dim'])
    except (TypeError,ValueError):pass
   if c.get('input_dim')!=3:errors.append(dict(name='input_dim',group='data',severity='error',message='PCCCompletion requires XYZ input.'))
  else:
   num('num_joints','Output joints',0,True,'model',True)
  try:c['input_dim']=int(c['input_dim'])
  except (TypeError,ValueError):errors.append(dict(name='input_dim',group='data',severity='error',message='Select XYZ, XYZ + Doppler, or XYZ + Doppler + SNR.'))
  if c.get('input_dim') not in (3,4,5) and not any(x['name']=='input_dim' for x in errors):errors.append(dict(name='input_dim',group='data',severity='error',message='Input features must be XYZ, XYZ + Doppler, or XYZ + Doppler + SNR.'))
  try:num('torch_threads','CPU threads',0,True,'runtime',True)
  except Exception:pass
  if str(c.get('device','cpu'))=='cpu':num('torch_threads','CPU threads',0,True,'runtime',True)
 def _resolve_checkpoint(self,value,errors,name='checkpoint',group='model',config=None):
  try:
   if not isinstance(value,str) or not value.strip():raise ValueError('Checkpoint is required.')
   path=Path(value).expanduser();path=(path if path.is_absolute() else self.project_root/path).resolve()
   if not path.is_relative_to(self.project_root) or not path.is_file():raise ValueError('Checkpoint must be an existing repository weight file.')
   if path.suffix.lower() not in ('.ckpt','.pt','.pth'):raise ValueError('Unsupported checkpoint type.')
   digest=_sha_file(path)
   if config is not None and __import__('importlib.util',fromlist=['find_spec']).find_spec('torch'):
    from mmwave_densepose.models import build_model
    from mmwave_densepose.engine.runner import load_weights
    settings={k:v for k,v in config.items() if k not in ('completion_checkpoint','_workbench','_public_data_root')}
    model=config['model'];model_name=config.get('model');model_object=build_model(model_name,settings);load_weights(model_object,path,model_name);del model_object
   return path,digest
  except (OSError,ValueError,TypeError,RuntimeError,ImportError) as exc:errors.append(exception_issue(exc,name=name,group=group));return None,None
 def _weight_id(self,model,checkpoint,digest):
  return checkpoint_log_dir(checkpoint,self.project_root).name if checkpoint_log_dir(checkpoint,self.project_root) else ('weights_'+digest[:12] if digest else None)
 def _bad(self,message,*,issue=None):
  if issue is None:issue=exception_issue(message) if isinstance(message,BaseException) else dict(name='configuration',group='model',severity='error',message=str(message))
  return {'report_id':'validation_'+uuid.uuid4().hex,'draft_hash':'','valid':False,'errors':[enrich_issue(issue,redact=_redact)],'warnings':[],'info':[]}
 def validate(self,draft):
  try:return self._validate(draft)
  except Exception as exc:return self._bad(exc)
 def _run_configuration(self,report):
  config=dict(report['config'])
  # PASTNet ablation selects its model architecture during testing too.
  internal=report.get('_config_internal',{})
  if config.get('model')=='PASTNet':config['ablation']=internal.get('ablation',config.get('ablation','full'))
  config['data_root']=str(self.data_root)
  return config
 @staticmethod
 def _run_options(report):
  signature=report['signature']
  return {'visual':bool(report.get('visual')),'save_predictions':bool(signature.get('save_predictions')),
          'allow_partial':bool(signature.get('allow_partial')),'limit_batches':signature.get('limit_batches')}
 def _command_preview(self,report):
  return ['python','-u',TASKS[report['action']]['entrypoint'],*explicit_run_arguments(
      report['action'],self._run_configuration(report),root=self.project_root,
      output_dir=report['output_dir'],**self._run_options(report))]
 def _job_copy(self,job):
  return {k:v for k,v in job.items() if k not in ('output_path','config_abs','process_args','process_config','env','pid','_output_path','_process_args','_process_config','_workbench_metadata','_layout_reservation')}
 def _save(self,job):
  output=deleted_logs.record_output(self.project_root,job)
  if output is not None and output.is_dir():job['log_published']=True
  _atomic_json(self._state_path(job['job_id']),self._job_copy(job))
 def _reserved_outputs(self):
  return {Path(j['_output_path']) for j in self._jobs.values() if j.get('status') in {'Queued','Starting','Running','Stopping'} and j.get('_output_path')}
 def submit(self,draft,report_id=None,request_id=None):
  if not isinstance(request_id,str) or not request_id or len(request_id)>MAX_REQUEST_ID or not re.fullmatch(r'[A-Za-z0-9_.:-]+',request_id):raise ValueError('A valid idempotency request ID is required.')
  self.prune_deleted_logs(force=True)
  with self._lock:
   old=self._requests.get(request_id)
   if old and old in self._jobs:return self._job_copy(self._jobs[old])
   stored=self._validations.get(report_id) if isinstance(report_id,str) else None
  if stored is None or not stored.get('valid'):raise ValueError('Run parameters must pass validation before submission.')
  current=self._validate(draft)
  if not current.get('valid'):raise ValueError('Run parameters changed or no longer pass validation.')
  if current['draft_hash']!=stored['draft_hash']:raise ValueError('Run parameters changed; validate them again.')
  # IDs shown in Review run are frozen. Reuse the original report's IDs even if
  # the same draft is submitted again after unrelated files appear.
  for key in ('run_id','weight_id','test_id','report_id'):
   current[key]=stored.get(key) if key!='report_id' else report_id
  out=Path(stored['output_path'])
  job_id='job_'+uuid.uuid4().hex[:24];execution_id='exec_'+uuid.uuid4().hex;now=time.time();run_id=stored.get('run_id')
  training_run_id=stored.get('training_run_id')
  workbench={'origin':'user','run_id':run_id or job_id,'job_id':job_id,'training_run_id':training_run_id,'weight_id':stored.get('weight_id'),'test_id':stored.get('test_id'),'visual':bool(stored.get('visual'))}
  config=self._run_configuration(stored)
  config_reference=save_run_config(stored['action'],config,root=self.project_root,**self._run_options(stored))
  command=self._command_preview(stored)
  args=[sys.executable,*command[1:]]
  with self._lock:
    if self._validations.get(report_id) is not stored:raise ValueError('Run parameters must pass validation before submission.')
    if request_id in self._requests:return self._job_copy(self._jobs[self._requests[request_id]])
    if out in self._reserved_outputs():raise ValueError('Output directory is already reserved by another job.')
    if out.exists():raise ValueError('Output directory already exists; existing results are never overwritten.')
    for old_job in self._jobs.values():
     if old_job.get('status') in {'Complete','Needs review'} and old_job.get('output_dir')==stored['output_dir']:
      raise ValueError('A prior job already owns this output directory. Choose a new run ID.')
    job={'job_id':job_id,'request_id':request_id,'report_id':report_id,'run_id':workbench['run_id'],'training_run_id':training_run_id,'weight_id':workbench.get('weight_id'),'test_id':stored.get('test_id'),'action':stored['action'],'model':config['model'],'status':'Queued','phase':'queued','progress':{'kind':'indeterminate','current':None,'total':None},'created_at':now,'updated_at':now,'started_at':None,'finished_at':None,'return_code':None,'error':None,'output_dir':stored['output_dir'],'config':stored['config'],'visual':bool(stored.get('visual')),'visual_pkl':None,'sample_count':next((int(re.match(r'(\d+)',i['message']).group(1)) for i in stored['info'] if i['name']=='split' and re.match(r'(\d+)',i['message'])),None),'log_cursor':0,'command_preview':command,'_output_path':str(out),'_process_args':args,'_process_config':config,'_workbench_metadata':workbench}
    job.update(config_reference)
    job['command_shell']=stored['command_shell']
    job['origin']='user';job['checkpoint_path']=stored.get('checkpoint_path');job['execution_id']=execution_id;job['launch_source']='gui'
    self._jobs[job_id]=job;self._requests[request_id]=job_id;self._cancel[job_id]=threading.Event();self._reserved.add(out);self._queue.append(job_id);self._save(job);self._wake.notify_all()
  return self._job_copy(job)
 def _update(self,job_id,**changes):
  with self._lock:
   job=self._jobs.get(job_id)
   if not job:return None
   job.update(changes,updated_at=time.time());self._save(job);return self._job_copy(job)
 def _append_log(self,job_id,stream,text):
  entry={'seq':0,'timestamp':time.time(),'stream':stream,'text':_redact(text)}
  with self._lock:
   job=self._jobs.get(job_id)
   if not job:return
   entry['seq']=int(job.get('log_cursor',0))+1;job['log_cursor']=entry['seq'];job['updated_at']=time.time()
   path=self._log_path(job_id);path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush()
   self._save(job)
 def logs(self,job_id,after=0,limit=500):
  with self._lock:
   job=self._jobs.get(job_id)
   if job is None:return None
   after=max(0,int(after));limit=max(1,min(int(limit),2000));path=self._log_path(job_id);lines=[]
   if path.is_file():
    with path.open(encoding='utf-8') as f:
     for line in f:
      try:item=json.loads(line)
      except (ValueError,TypeError):continue
      if item.get('seq',0)>after:lines.append(item)
      if len(lines)>=limit:break
   nxt=lines[-1]['seq'] if lines else after
   return {'lines':lines,'next':nxt,'caught_up':nxt>=job.get('log_cursor',0)}
 def status(self,job_id):
  self.prune_deleted_logs()
  with self._lock:
   job=self._jobs.get(job_id);return None if job is None else self._job_copy(job)
 def list_jobs(self):
  self.prune_deleted_logs()
  with self._lock:return [self._job_copy(job) for job in sorted(self._jobs.values(),key=lambda j:j.get('created_at',0),reverse=True)[:200]]
 def manages_job(self,job_id):
  """Only this runner instance may control its queued or launched children."""
  with self._lock:return job_id in self._cancel
 def stop(self,job_id):
  with self._lock:
   job=self._jobs.get(job_id)
   if job is None:return None
   if job['status']=='Queued':
    self._queue=deque(item for item in self._queue if item!=job_id);job.update(status='Stopped',phase='stopped',finished_at=time.time(),error=None,updated_at=time.time());self._save(job);self._release(job_id);return self._job_copy(job)
   if job['status'] not in ('Starting','Running','Stopping'):return self._job_copy(job)
   job.update(status='Stopping',phase='stopping',updated_at=time.time());self._save(job);self._cancel[job_id].set();proc=self._procs.get(job_id)
  if proc is not None:
   try:
    if os.name=='nt' and hasattr(signal,'CTRL_BREAK_EVENT'):proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:proc.terminate()
   except (OSError,ProcessLookupError):pass
   threading.Thread(target=self._force_stop_after_grace,args=(job_id,proc),name=f'workbench-stop-{job_id}',daemon=True).start()
  return self.status(job_id)
 def _force_stop_after_grace(self,job_id,proc):
  try:proc.wait(timeout=8)
  except subprocess.TimeoutExpired:
   try:
    if os.name=='nt':
     subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,shell=False)
    else:os.killpg(proc.pid,signal.SIGKILL)
   except (OSError,ProcessLookupError):
    try:proc.kill()
    except (OSError,ProcessLookupError):pass
 def _dispatch(self):
  while True:
   with self._wake:
    while not self._queue and not self._closed:self._wake.wait()
    if self._closed:return
    job_id=self._queue.popleft();job=self._jobs.get(job_id)
    if not job or job['status']!='Queued':continue
    job.update(status='Starting',phase='starting',started_at=time.time(),updated_at=time.time());self._save(job)
   self._run_job(job_id)
 def _read_stream(self,job_id,name,stream):
  try:
   for line in iter(stream.readline,''):
    self._append_log(job_id,name,line)
    self._consume_progress(job_id,line)
  except (OSError,ValueError):pass
  finally:
   try:stream.close()
   except OSError:pass
 def _consume_progress(self,job_id,line):
  try:row=json.loads(line)
  except (ValueError,TypeError):row=None
  progress=None
  with self._lock:
   job=self._jobs.get(job_id)
   if not job:return
   action=job['action']
   if action=='train' and isinstance(row,dict) and isinstance(row.get('epoch'),int):
    progress={'kind':'epoch','current':row['epoch'],'total':int(job['config'].get('epochs',200)),'train_loss':row.get('train_loss'),
              'val_metric':row.get('val_cd_m',row.get('val_mle_m'))}
    job.update(phase='training',status='Running')
   else:
    match=re.search(r'Evaluated\s+(\d+)\s+samples',line)
    if match and job['action'] in ('evaluate','validate','infer'):
     total=job.get('sample_count')
     progress={'kind':'samples','current':int(match.group(1)),'total':total}
     job.update(phase='evaluating',status='Running')
   if progress:
    job['progress']=progress;job['updated_at']=time.time();self._save(job)
 def _run_job(self,job_id):
  with self._lock:
   job=self._jobs.get(job_id)
   if not job:return
   args=list(job['_process_args']);env=dict(os.environ);env['PYTHONUNBUFFERED']='1'
   env['MMWAVE_EXECUTION_ID']=job['execution_id'];env['MMWAVE_OWNER_JOB_ID']=job_id
   env['MMWAVE_WORKBENCH_METADATA']=json.dumps(job.get('_workbench_metadata',{}),allow_nan=False)
   env['MMWAVE_PUBLIC_DATA_ROOT']=str(job['config'].get('data_root','configured-data-root'))
   env['MMWAVE_RUN_CONFIG_FILE']=job['config_file'];env['MMWAVE_RUN_CONFIG_ID']=job['config_id'];env['MMWAVE_RUN_CONFIG_SHA256']=job['config_sha256']
   if self._cancel.get(job_id,threading.Event()).is_set():
    self._update(job_id,status='Stopped',phase='stopped',finished_at=time.time(),error=None);self._release(job_id);return
  try:
   proc=subprocess.Popen(args,cwd=str(self.project_root),env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
       text=True,encoding='utf-8',errors='replace',bufsize=1,shell=False,
       **({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}))
  except (OSError,ValueError) as exc:
   self._append_log(job_id,'stderr',str(exc));self._update(job_id,status='Failed',phase='failed',finished_at=time.time(),error=_redact(str(exc)));self._release(job_id);return
  with self._lock:
   self._procs[job_id]=proc;job=self._jobs[job_id];job.update(pid=proc.pid,status='Running',phase='initializing',updated_at=time.time());self._save(job)
  readers=[threading.Thread(target=self._read_stream,args=(job_id,name,stream),daemon=True) for name,stream in (('stdout',proc.stdout),('stderr',proc.stderr))]
  for reader in readers:reader.start()
  code=proc.wait()
  for reader in readers:reader.join(timeout=3)
  with self._lock:cancelled=self._cancel.get(job_id,threading.Event()).is_set();self._procs.pop(job_id,None)
  if cancelled:
   self._update(job_id,status='Stopped',phase='stopped',return_code=code,finished_at=time.time(),error=None)
  elif code!=0:
   self._update(job_id,status='Failed',phase='failed',return_code=code,finished_at=time.time(),error=f'Process exited with code {code}. See console output.')
  else:
   self._finish_success(job_id,code)
  self._release(job_id)
 def _finish_success(self,job_id,code):
  with self._lock:job=dict(self._jobs[job_id])
  out=Path(job['_output_path']);action=job['action'];expected=[out/'run.json',out/'hparams.yaml',out/'metrics.csv']
  if action=='train':
   checkpoint=job.get('checkpoint_path')
   expected += [out/'training_summary.json',self.project_root/checkpoint if checkpoint else self.artifact_roots['checkpoints']/job['model']/str(job.get('run_id') or out.name)/'best.ckpt']
  else:
   expected += [out/'metrics.json',out/('per_cloud_metrics.csv' if job['model']==COMPLETION_MODEL else 'per_joint_metrics.csv')]
   if job['model']!=COMPLETION_MODEL:expected.append(out/'per_joint_mle.csv')
  missing=[p.name for p in expected if not p.is_file()]
  if not any(p.is_file() and p.stat().st_size>0 for p in out.glob('events.out.tfevents.*')):missing.append('TensorBoard events')
  visual_rel=None
  if job.get('visual'):
   visual=out/'test_results.pkl';metadata=out/'run.json'
   if not visual.is_file() or visual.stat().st_size==0:missing.append('test_results.pkl')
   elif metadata.is_file():
    try:
     data=json.loads(metadata.read_text(encoding='utf-8'));claimed=data.get('visual_pkl_sha256')
     if not data.get('visual') or data.get('visual_pkl')!='test_results.pkl' or not claimed or _sha_file(visual)!=claimed:missing.append('test_results.pkl integrity')
     visual_rel=visual.relative_to(self.project_root).as_posix()
    except (OSError,ValueError,TypeError):missing.append('visual PKL metadata')
   else:missing.append('visual PKL metadata')
  if missing:self._update(job_id,status='Failed',phase='missing-artifacts',return_code=code,finished_at=time.time(),error='Expected artifacts are missing: '+', '.join(missing)+'.')
  else:self._update(job_id,status='Complete',phase='complete',progress={'kind':'complete','current':1,'total':1},return_code=code,finished_at=time.time(),visual_pkl=visual_rel,error=None)
 def _release(self,job_id):
  with self._lock:
   job=self._jobs.get(job_id)
   if job:
    self._reserved.discard(Path(job['_output_path']))
   self._wake.notify_all()
 def close(self):
  with self._lock:
   self._closed=True
   queued=list(self._queue);self._queue.clear()
   for job_id in queued:
    job=self._jobs.get(job_id)
    if job and job.get('status')=='Queued':job.update(status='Stopped',phase='stopped',finished_at=time.time(),error=None,updated_at=time.time());self._save(job)
   active=[k for k,j in self._jobs.items() if j.get('status') in {'Starting','Running','Stopping'}]
   self._wake.notify_all()
  for job_id in active:self.stop(job_id)
