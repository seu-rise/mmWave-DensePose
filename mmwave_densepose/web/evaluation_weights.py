"""Repository-owned evaluation weights and their immutable training settings."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
import re
from pathlib import Path
import stat
import threading

import yaml

from mmwave_densepose.data.partitions import SINGLE_MODELS,normalize_data_config
from mmwave_densepose.engine.run_config import default_run_config
from mmwave_densepose.workflows.catalog import list_registered_models


CONFIG_FIELDS = frozenset({"model", "task", "data_root", "split_strategy", "partition", "seed",
    "split_chunk_size", "min_valid_points", "max_timestamp_gap_ms", "num_points", "input_dim", "seq_len",
    "num_joints", "ablation", "num_out_points", "pcc_label_radius", "dense_label_mode", "completion_checkpoint",
    "batch_size", "device", "gpu_ids", "num_workers", "torch_threads", "cache_size", "epochs", "learning_rate",
    "weight_decay", "optimizer", "scheduler_name", "scheduler_eta_min", "scheduler_step_size", "scheduler_gamma", "checkpoint"})
RUNTIME_FIELDS = frozenset({"batch_size", "device", "gpu_ids", "num_workers", "torch_threads", "cache_size"})
EDITABLE_FIELDS = sorted(RUNTIME_FIELDS | {"visual", "save_predictions"})
SEMANTIC_FIELDS = frozenset({"model","task","split_strategy","partition","seed","split_chunk_size",
    "min_valid_points","max_timestamp_gap_ms","num_points","input_dim","seq_len","num_joints","ablation",
    "num_out_points","pcc_label_radius","dense_label_mode"})
ALIASES = {"radar_point_size":"num_points", "sample_dim":"input_dim", "feature_dim":"input_dim",
           "num_keypoints":"num_joints", "t_max":"seq_len", "max_epochs":"epochs",
           "effective_max_epochs":"epochs", "optimizer_name":"optimizer"}
MODEL_ALIASES = {"PartTokenFormer":"PASTNet", "mmPoseFK":"MMPoseFK", "PCC":"PCCCompletion", "PCCN":"GlobalLocalNetwork"}


class EvaluationWeightError(ValueError):
    def __init__(self,message,*,name='checkpoint',group='model',code='evaluation_weight_invalid',
                 suggestion='Refresh the weight list and select a compatible training checkpoint.',**context):
        super().__init__(message)
        self.issue=dict(name=name,group=group,severity='error',message=message,
                        code=code,suggestion=suggestion,**context)


def _parameter_group(name):
    if name in {'data_root','split_strategy','partition','seed','split_chunk_size','min_valid_points',
                'max_timestamp_gap_ms','num_points','input_dim','seq_len','pcc_label_radius'}:return 'data'
    if name in RUNTIME_FIELDS or name in {'limit_batches','allow_partial'}:return 'runtime'
    if name.startswith('scheduler_'):return 'schedule'
    if name in {'epochs','learning_rate','weight_decay','optimizer'}:return 'training'
    return 'model'


def safe_path(root, value):
    root=Path(root).resolve();path=Path(value);path=path if path.is_absolute() else root/path
    try: relative=path.relative_to(root)
    except ValueError: raise EvaluationWeightError("Weight configuration path is outside the repository") from None
    current=root
    for part in relative.parts:
        if part in (".",".."):raise EvaluationWeightError("Weight configuration path is invalid")
        current/=part
        try: info=current.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400:
            raise EvaluationWeightError("Weight configuration paths must not contain links")
    resolved=path.resolve()
    if not resolved.is_relative_to(root):raise EvaluationWeightError("Weight path escaped the repository")
    return resolved


def _mapping(root,path):
    try:
        path=safe_path(root,path)
        if not path.is_file() or not 0<path.stat().st_size<=2*1024*1024:return {}
        value=yaml.safe_load(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,TypeError,yaml.YAMLError):return {}


def canonical(values):
    result={}
    if not isinstance(values,dict):return result
    for key,value in (values or {}).items():
        name=ALIASES.get(key,key)
        if name not in CONFIG_FIELDS:continue
        if isinstance(value,str) and value.strip() in ('None','null',''):value=None
        if value is None and name!='split_chunk_size':continue
        if name in result and key!=name:continue
        if name=='partition' and isinstance(value,str):
            try:value=ast.literal_eval(value)
            except (ValueError,SyntaxError):continue
        if isinstance(value,tuple):value=list(value)
        if name=='model':value=MODEL_ALIASES.get(value,value)
        if name=='task':value={'keypoint':'pose','keypoint_muti':'pose','dense':'completion'}.get(value,value)
        if value is None or isinstance(value,(str,bool,int,float,list)):
            result[name]=copy.deepcopy(value)
    if 'split_strategy' not in result and isinstance((values or {}).get('split_manifest'),str):
        split=values['split_manifest']
        if 'subject' in split:result['split_strategy']='subject'
        elif 'intra' in split:result['split_strategy']='intra_sequence'
        elif 'chunk' in split:result['split_strategy']='sequence_chunk'
    return result


def _snapshot_reference(root,metadata):
    reference=metadata.get('config_file');expected=metadata.get('config_sha256')
    if not isinstance(reference,str) or not isinstance(expected,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',expected):return None
    try:
        path=safe_path(root,reference)
        if path.is_relative_to(Path(root)/'configs/runs'):return path,expected.lower()
    except (OSError,ValueError):pass
    return None


class EvaluationWeights:
    def __init__(self,root,data_root):
        self.root=Path(root).resolve();self.data_root=Path(data_root).resolve();self._cache={};self._lock=threading.RLock()

    def _public_data_root(self):
        return self.data_root.relative_to(self.root).as_posix() if self.data_root.is_relative_to(self.root) else str(self.data_root)

    def _checkpoint_config(self,path):
        """Inspect safe tensor metadata without materializing model tensor storage."""
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(checkpoint,dict):return {},None
        config=checkpoint.get('config');kind='resolved'
        if not isinstance(config,dict) or not config:config=checkpoint.get('hyper_parameters');kind='legacy'
        if not isinstance(config,dict):return {},None
        result=canonical(config)
        model=checkpoint.get('model_name',checkpoint.get('model'))
        if isinstance(model,str):result['model']=MODEL_ALIASES.get(model,model)
        return result,kind

    def _training_config(self,directory):
        values={};sources=[]
        for filename in ('hparams.yaml','hparams.yml','paper-training.json','run.json'):
            path=directory/filename;metadata=_mapping(self.root,path)
            if not metadata:continue
            raw=(metadata.get('config') or {}) if filename.endswith('.json') else metadata
            if filename=='paper-training.json':values.update(canonical(metadata.get('paper_configuration') or {}))
            values.update(canonical(raw));sources.append(path.relative_to(self.root).as_posix())
            reference=_snapshot_reference(self.root,metadata)
            if reference:
                try:
                    snapshot,expected=reference
                    if (snapshot.is_file() and 0<snapshot.stat().st_size<=2*1024*1024
                            and hashlib.sha256(snapshot.read_bytes()).hexdigest()==expected):
                        values.update(canonical(_mapping(self.root,snapshot)));sources.append(snapshot.relative_to(self.root).as_posix())
                except (OSError,ValueError):pass
        return values,sources

    def _entries(self,model):
        manifest=_mapping(self.root,self.root/'paper/manifest.json')
        for asset in manifest.get('assets',[]) if isinstance(manifest.get('assets',[]),list) else []:
            if not isinstance(asset,dict) or asset.get('model')!=model or not isinstance(asset.get('id'),str):continue
            path=asset.get('path')
            if not isinstance(path,str):continue
            yield {'id':'official:'+asset['id'],'model':model,'source':'official','path':path,
                   'training_run_id':asset.get('training_run_id') or asset['id'],'asset':asset}
        base=self.root/'mmwave_densepose/checkpoints'/model
        try:
            base=safe_path(self.root,base)
            children=sorted(base.iterdir()) if base.is_dir() else []
        except (OSError,ValueError):children=[]
        for directory in children:
            if not directory.is_dir():continue
            for filename in ('best.ckpt','last.ckpt','best.pt','best.pth'):
                path=directory/filename
                if not path.is_file():continue
                relative=path.relative_to(self.root).as_posix()
                yield {'id':'user:'+hashlib.sha256(relative.encode()).hexdigest()[:24],'model':model,'source':'user',
                       'path':relative,'training_run_id':directory.name,'asset':{}}

    def _descriptor(self,entry):
        asset=entry['asset'];path=safe_path(self.root,entry['path'])
        if entry['source']=='official':
            allowed=self.root/'paper/experiments'/asset['id']/'checkpoints'
            if path.parent!=allowed:raise EvaluationWeightError('Official weight is outside its registered asset')
            directory=safe_path(self.root,asset.get('training_log') or 'mmwave_densepose/logs/train/'+entry['model']+'/'+entry['training_run_id'])
        else:
            allowed=self.root/'mmwave_densepose/checkpoints'/entry['model']/entry['training_run_id']
            if path.parent!=allowed:raise EvaluationWeightError('User weight is outside its registered training run')
            directory=self.root/'mmwave_densepose/logs/train'/entry['model']/entry['training_run_id']
        descriptor={k:entry[k] for k in ('id','model','source','training_run_id')}
        descriptor['asset_id'] = asset.get('id') if entry['source'] == 'official' else None
        descriptor.update(checkpoint=path.relative_to(self.root).as_posix(),label=f"{entry['training_run_id']} / {path.name}",
                          config={},locked_fields=[],editable_fields=EDITABLE_FIELDS,available=False,reason='',paper_tables=asset.get('paper_tables',[]),compatibility_notes=[])
        if not path.is_file():descriptor['reason']='Checkpoint is not installed';return descriptor
        stamps=[(str(path),path.stat().st_size,path.stat().st_mtime_ns)]
        for candidate in [self.root/'paper/manifest.json',*(directory/name for name in ('run.json','paper-training.json','hparams.yaml','hparams.yml'))]:
            try:stamps.append((str(candidate),candidate.stat().st_size,candidate.stat().st_mtime_ns))
            except OSError:pass
            reference=_snapshot_reference(self.root,_mapping(self.root,candidate))
            if reference:
                snapshot,expected=reference
                try:info=snapshot.stat();stamps.append((str(snapshot),info.st_size,info.st_mtime_ns,expected))
                except OSError:stamps.append((str(snapshot),None,None,expected))
        config_path=asset.get('config')
        if isinstance(config_path,str):
            candidate=safe_path(self.root,config_path)
            try:stamps.append((str(candidate),candidate.stat().st_size,candidate.stat().st_mtime_ns))
            except OSError:pass
        signature=tuple(stamps)
        with self._lock:
            cached=self._cache.get(entry['id'])
            if cached and cached[0]==signature:return copy.deepcopy(cached[1])
        values={};sources=[]
        if isinstance(config_path,str):
            values.update(canonical(_mapping(self.root,config_path)));sources.append(config_path)
        logged,logged_sources=self._training_config(directory);values.update(logged);sources.extend(logged_sources)
        if entry['source']=='user':
            try:
                embedded,kind=self._checkpoint_config(path)
                if embedded:
                    conflicts=sorted(name for name in SEMANTIC_FIELDS if name in embedded and name in values
                        and embedded[name]!=values[name] and not (name=='seq_len' and entry['model'] in SINGLE_MODELS))
                    if kind=='resolved':
                        if conflicts:descriptor['compatibility_notes'].append('Checkpoint-saved resolved configuration takes precedence over training log values for: '+', '.join(conflicts))
                        values.update(embedded)
                    else:
                        if conflicts:descriptor['reason']='Saved training configuration conflicts with checkpoint hyperparameters: '+', '.join(conflicts)
                        values={**embedded,**values}
                    sources.append(descriptor['checkpoint']+('#config' if kind=='resolved' else '#hyper_parameters'))
            except (OSError,ValueError,RuntimeError,ImportError,TypeError,KeyError,pickle.UnpicklingError):
                if not logged:descriptor['reason']='Training configuration could not be recovered from this checkpoint'
        if values.get('model') not in (None,entry['model']):descriptor['reason']='Training configuration model disagrees with the weight'
        values['model']=entry['model']
        if entry['model'] in SINGLE_MODELS:
            # Legacy temporal data loaders could record a multi-frame window for
            # a single-frame model. The registered model consumes one frame.
            values['seq_len']=1;sources.append('model data family: single_frame')
        required={'num_points','input_dim','seq_len','seed','split_strategy','partition','min_valid_points','max_timestamp_gap_ms'}
        if values.get('split_strategy')=='sequence_chunk':required.add('split_chunk_size')
        if entry['model']!='PCCCompletion':required.add('num_joints')
        if entry['model']=='PASTNet':required.add('ablation')
        if entry['model']=='PCCCompletion':required.update(('num_out_points','pcc_label_radius'))
        missing=sorted(name for name in required if name not in values or values[name] is None)
        if missing:descriptor['reason']='Training configuration is missing: '+', '.join(missing)
        if values.get('split_strategy') not in (None,'sequence_chunk','subject'):
            descriptor['reason']='Training uses an unsupported legacy split protocol: '+str(values['split_strategy'])+'. Its original partition must be recovered before evaluation.'
        if not descriptor['reason']:
            try:
                if entry['model']=='GlobalLocalNetwork':values.setdefault('num_out_points',196)
                values={**default_run_config(entry['model'],device='cpu'),**values}
                values['data_root']=self._public_data_root();values['checkpoint']=descriptor['checkpoint']
                values=normalize_data_config(values)
                values={key:value for key,value in values.items() if key in CONFIG_FIELDS}
                descriptor.update(config=values,locked_fields=sorted(set(values)-RUNTIME_FIELDS),available=True,
                                  config_sources=sources,weight_signature=hashlib.sha256(repr(signature).encode()).hexdigest())
            except (ValueError,TypeError,KeyError) as exc:descriptor['reason']='Training configuration is invalid: '+str(exc)
        with self._lock:self._cache[entry['id']]=(signature,copy.deepcopy(descriptor))
        return descriptor

    def list_weights(self,model):
        if model not in list_registered_models():raise EvaluationWeightError('Select a registered model')
        weights=[]
        for entry in self._entries(model):
            try:weights.append(self._descriptor(entry))
            except (OSError,ValueError,TypeError):continue
        return {'model':model,'weights':weights}

    def bind(self,action,provided,options):
        """Bind explicit IDs and known legacy checkpoint paths; reject locked edits."""
        if action=='train':return dict(provided),None
        identifier=options.get('evaluation_weight_id');model=provided.get('model')
        if identifier is not None and not isinstance(identifier,str):raise EvaluationWeightError('Select a valid evaluation weight')
        entries=self.list_weights(model)['weights'] if model in list_registered_models() else []
        selected=next((item for item in entries if item['id']==identifier),None) if identifier else None
        if not identifier and provided.get('checkpoint'):
            try:checkpoint=safe_path(self.root,provided['checkpoint']).relative_to(self.root).as_posix()
            except (OSError,ValueError,TypeError):raise EvaluationWeightError('Checkpoint paths must stay inside the repository without linked entries') from None
            selected=next((item for item in entries if item['checkpoint']==checkpoint),None)
        if selected is None:
            if identifier:raise EvaluationWeightError('Evaluation weight is unavailable for the selected model')
            return dict(provided),None  # Compatibility for non-indexed legacy clients.
        if not selected['available']:raise EvaluationWeightError(selected['reason'] or 'Training configuration is unavailable')
        if options.get('limit_batches') not in (None,'',0,'0') or options.get('allow_partial') not in (None,False):
            raise EvaluationWeightError('Selected-weight evaluation requires the complete dataset and split',
                name='allow_partial' if options.get('allow_partial') else 'limit_batches',group='runtime',
                code='evaluation_full_split_required',suggestion='Restore the selected weight settings and evaluate the complete recorded split.',
                action='refresh_weights')
        expected=selected['config']
        for name,value in provided.items():
            if name in RUNTIME_FIELDS:continue
            if name not in expected:raise EvaluationWeightError('Parameter is not available for the selected weight: '+name,
                name=name,group=_parameter_group(name),code='evaluation_weight_config_stale',actual=value,
                suggestion='Restore the selected weight settings to remove parameters unavailable in this training record.',action='refresh_weights')
            wanted=expected[name]
            if name=='data_root':
                try:
                    actual=Path(value).expanduser();actual=(actual if actual.is_absolute() else self.root/actual).resolve()
                    equal=actual==self.data_root
                except (OSError,ValueError,TypeError):equal=False
            elif name in ('checkpoint','completion_checkpoint') and value not in (None,''):
                try:equal=safe_path(self.root,value)==safe_path(self.root,wanted)
                except (OSError,ValueError,TypeError):equal=False
            elif isinstance(wanted,(int,float)) and not isinstance(wanted,bool):
                try:equal=not isinstance(value,bool) and float(value)==float(wanted)
                except (TypeError,ValueError):equal=False
            else:equal=value==wanted
            if not equal:raise EvaluationWeightError('Parameter is locked to the selected training weight: '+name,
                name=name,group=_parameter_group(name),code='evaluation_parameter_locked',actual=value,expected=wanted,
                suggestion='Restore the selected weight settings. To use different model or input settings, select a weight trained with those settings.',
                action='refresh_weights')
        bound=copy.deepcopy(expected)
        bound.update({name:copy.deepcopy(value) for name,value in provided.items() if name in RUNTIME_FIELDS})
        options['evaluation_weight_id']=selected['id']
        return bound,selected
