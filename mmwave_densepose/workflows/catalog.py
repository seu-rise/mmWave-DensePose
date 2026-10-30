"""Model and task catalogue read from the trusted package registry and config files."""
from __future__ import annotations
import ast
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'mmwave_densepose' / 'models' / '__init__.py'
COMPLETION_MODEL = 'PCCCompletion'
TASKS = {
    'train': {'label': 'Train', 'entrypoint': 'scripts/train.py'},
    'validate': {'label': 'Validate', 'entrypoint': 'scripts/validate.py'},
    'evaluate': {'label': 'Test', 'entrypoint': 'scripts/evaluate.py'},
    'infer': {'label': 'Predict', 'entrypoint': 'scripts/infer.py'},
}

def list_registered_models():
    tree = ast.parse(REGISTRY.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_MODELS' for t in node.targets):
            registry = ast.literal_eval(node.value)
            if isinstance(registry, dict) and registry and all(isinstance(k, str) for k in registry):
                return list(registry)
    raise ValueError('The model registry does not expose a static _MODELS mapping.')

def model_slug(model):
    return {'PASTNet':'pastnet','PointNet2Baseline':'pointnet2','DGCNN':'dgcnn',
            'PointTransformer':'point-transformer','PointMLP':'pointmlp','MMPoseFK':'mm-pose-fk',
            'DualPathCNNBiLSTM':'dual-path','PntPoseAT':'pnt-pose-at','PoseGraphNet':'pose-graph',
            'GlobalLocalNetwork':'global-local','PCCCompletion':'pcc'}.get(model,model.lower().replace(' ','-'))

@lru_cache(maxsize=1)
def get_catalog():
    models = list_registered_models(); templates = []
    for path in sorted((ROOT/'paper'/'configs').glob('*.json')):
        try: config=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError,TypeError): continue
        if config.get('model') not in models: continue
        templates.append({'id':path.stem,'label':path.stem.removeprefix('main-').replace('-',' ').title(),
                          'model':config['model'],'path':path.relative_to(ROOT).as_posix(),'config':config})
    from mmwave_densepose.data.partitions import normalize_data_config
    profiles = {m:normalize_data_config({'model':m}) for m in models}
    return {'schema_version':2,'split_strategies':['sequence_chunk','subject'],'actions':{k:dict(v,id=k) for k,v in TASKS.items()},
            'models':[{'id':m,'task':'completion' if m==COMPLETION_MODEL else 'pose',
                       'ablations':['full','spatial_only','baseline'] if m=='PASTNet' else [],
                       'visual':m!=COMPLETION_MODEL,'data_family':profiles[m]['data_family'],
                       'default_seq_len':profiles[m]['seq_len']} for m in models],'templates':templates}

@lru_cache(maxsize=4)
def _official_log_map(manifest_path, modified, size):
    """Only the prepared paper manifest designates official log directories."""
    del modified, size
    path = Path(manifest_path)
    root = path.parent.parent
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
    result = {}
    assets = manifest.get('assets', [])
    if not isinstance(assets, list):
        return result
    for asset in assets:
        if (not isinstance(asset, dict) or not isinstance(asset.get('id'), str)
                or not isinstance(asset.get('model'), str) or not asset['id'] or not asset['model']
                or any(part in asset['model'] for part in ('/', '\\', '..'))):
            continue
        entries = [('train', asset.get('training_log'))]
        entries.extend(('test', item.get('path') if isinstance(item, dict) else item)
                       for item in asset.get('evaluations', []))
        for phase, relative in entries:
            if not isinstance(relative, str):
                continue
            candidate = Path(relative)
            if candidate.is_absolute() or '..' in candidate.parts:
                continue
            directory = (root / candidate).resolve()
            allowed = (root / 'mmwave_densepose' / 'logs' / phase / asset['model']).resolve()
            if directory == allowed or not directory.is_relative_to(allowed):
                continue
            result[(phase, str(directory))] = asset
    return result


def official_asset_for_log(directory, repository_root, *, phase, metadata):
    """Authenticate path, asset and model together; origin text is not authority."""
    manifest = Path(repository_root).resolve() / 'paper/manifest.json'
    try:
        if manifest.is_symlink():
            return None
        stat = manifest.stat()
        asset = _official_log_map(str(manifest), stat.st_mtime_ns, stat.st_size).get(
            (phase, str(Path(directory).resolve())))
    except (OSError, ValueError, TypeError):
        return None
    if not asset or not isinstance(metadata, dict):
        return None
    return asset if metadata.get('asset_id') == asset['id'] and metadata.get('model') == asset['model'] else None


def task_type_for(model):
    if model not in list_registered_models(): raise ValueError(f'Unknown model: {model}')
    return 'completion' if model==COMPLETION_MODEL else 'pose'

def action_for_model(model,action):
    if action not in TASKS: raise ValueError(f'Unknown action: {action}')
    return task_type_for(model)

__all__=['COMPLETION_MODEL','ROOT','TASKS','action_for_model','get_catalog','list_registered_models','model_slug','task_type_for','official_asset_for_log']
