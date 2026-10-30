"""Freeze manuscript references and compatible local test bests for the paper GUI.

Only JSON metadata and CSV metrics are read. No dataset PKL or model is loaded.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

MAIN = ['main-dgcnn', 'main-point-transformer', 'main-pointmlp', 'main-mm-pose-fk',
        'main-dual-path', 'main-pnt-pose-at', 'main-pose-graph', 'main-global-local', 'main-pastnet']
MODELS = ['DGCNN', 'PointTransformer', 'PointMLP', 'MMPoseFK', 'DualPathCNNBiLSTM',
          'PntPoseAT', 'PoseGraphNet', 'GlobalLocalNetwork', 'PASTNet']
MODEL_EXPERIMENT = dict(zip(MODELS, MAIN))
TITLES = ['Data streams', 'Dataset statistics', 'Related datasets', 'Overall performance',
          'Per-joint MLE', 'Five-seed robustness', 'Split protocols', 'Module ablation',
          'Input features', 'Point count ablation']
METRICS = [('mle_cm', 'min'), ('pa_mpjpe_cm', 'min'), ('pck_percent', 'max'), ('auc_percent', 'max')]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def number(value):
    try:
        n = float(str(value).strip())
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def text(value):
    return ' '.join(str(value).split())


def relative(path, root):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def safe_inside(path, root):
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def split_name(value):
    return Path(str(value)).stem


def fingerprint(config):
    protocol = (('strategy', config['split_strategy']) if 'split_strategy' in config else
                ('legacy-index', split_name(config.get('split_manifest', 'chunk_s42'))))
    return (config.get('model'), protocol,
            int(config.get('seed', 42)), int(config.get('num_points', 300)),
            int(config.get('input_dim', 5)), int(config.get('seq_len', 5)),
            config.get('ablation', 'full'), tuple(config.get('partition', (.8, .1, .1))),
            config.get('split_chunk_size', 200), config.get('max_timestamp_gap_ms', 300),
            config.get('min_valid_points', 50))


def current_identity(root, config):
    data = Path(config.get('data_root', 'data/mmwave_densepose'))
    if not data.is_absolute():
        data = root / data
    if not safe_inside(data, root):
        return None
    manifest = data / 'manifest.json'
    source_manifest = manifest
    if not manifest.is_file():
        return None
    document = read_json(manifest)
    if 'split_strategy' in config:
        try:
            from mmwave_densepose.data.partitions import inspect_partition
            current = inspect_partition(data, config)
            metadata = current['metadata']
            if current['pending'] or metadata.get('partial') or document.get('partial'):
                return None
            split = Path(current['metadata_path'])
            return {'manifest': relative(source_manifest, root), 'manifest_sha256': digest(source_manifest),
                    'split': relative(split, root), 'split_sha256': digest(split), 'metadata': metadata,
                    'cache_id': current['cache_id']}
        except (OSError, ValueError, TypeError, KeyError):
            return None
    if document.get('dataset_kind') == 'full_multimodal':
        data = data / 'cache' / 'pose-v1'
        manifest = data / 'manifest.json'
    split = data / 'splits' / (split_name(config.get('split_manifest', 'chunk_s42')) + '.json')
    if not manifest.is_file() or not split.is_file():
        return None
    metadata = read_json(split)
    index_file = split.with_suffix('.npz')
    if metadata.get('index_sha256') and (not index_file.is_file() or digest(index_file) != metadata['index_sha256']):
        return None
    if metadata.get('partial') or read_json(manifest).get('partial'):
        return None
    return {'manifest': relative(source_manifest, root), 'manifest_sha256': digest(source_manifest),
            'split': relative(split, root), 'split_sha256': digest(split), 'metadata': metadata}


def normalized_metrics(metrics):
    out = dict(metrics)
    for source, target, scale in [('mle_m', 'mle_cm', 100), ('pa_mpjpe_m', 'pa_mpjpe_cm', 100),
                                  ('pck', 'pck_percent', 100), ('auc', 'auc_percent', 100)]:
        if target not in out and finite(out.get(source)):
            out[target] = out[source] * scale
    return out


def compatible_record(run, metrics, config, identity):
    if not identity:
        return 'current_dataset_identity_unavailable'
    if run.get('action') not in ('evaluate', 'test') or run.get('split') != 'test' or metrics.get('split') != 'test':
        return 'not_test_evaluation'
    if run.get('partial') or metrics.get('partial') or metrics.get('dataset_partial'):
        return 'partial_dataset'
    if run.get('limit_batches') is not None or metrics.get('limit_batches') is not None:
        return 'limited_run'
    if metrics.get('evaluation_complete') is False:
        return 'incomplete_evaluation'
    if run.get('model') != config.get('model') or metrics.get('model', config.get('model')) != config.get('model'):
        return 'model_identity_mismatch'
    if 'seed' in run and int(run['seed']) != int(config.get('seed', 42)):
        return 'seed_identity_mismatch'
    if 'split_strategy' in config:
        if run.get('split_strategy', run.get('config', {}).get('split_strategy')) != config['split_strategy']:
            return 'split_identity_mismatch'
        if run.get('partition_cache_id') not in (None, identity.get('cache_id')):
            return 'split_cache_identity_mismatch'
    elif run.get('split_manifest') and split_name(run['split_manifest']) != split_name(config.get('split_manifest', 'chunk_s42')):
        return 'split_identity_mismatch'
    if fingerprint(run.get('config', {})) != fingerprint(config):
        return 'configuration_mismatch'
    if run.get('dataset_manifest_sha256') != identity['manifest_sha256']:
        return 'dataset_identity_mismatch'
    metadata, expected = run.get('split_metadata', {}), identity['metadata']
    for key in ('seq_len', 'split_strategy', 'source_index_sha256', 'index_sha256', 'split_num_samples'):
        if key in expected and metadata.get(key) != expected[key]:
            return 'frozen_split_mismatch'
    samples = expected.get('split_num_samples', {}).get('test')
    if not isinstance(samples, int) or samples <= 0 or metrics.get('samples') != samples:
        return 'sample_population_mismatch'
    if metrics.get('expected_samples', samples) != samples or metrics.get('evaluated_samples', samples) != samples:
        return 'incomplete_sample_population'
    if metrics.get('threshold_m') != 0.15:
        return 'threshold_mismatch'
    joints = 17 if config['model'] == 'MMPoseFK' else 19
    if metrics.get('joints') != joints:
        return 'joint_set_mismatch'
    for name, _ in METRICS:
        value = metrics.get(name)
        if finite(value) and (value < 0 or (name in ('pck_percent', 'auc_percent') and value > 100)):
            return 'invalid_metric_range'
    if not any(finite(metrics.get(name)) for name, _ in METRICS):
        return 'no_scalar_metrics'
    return None


def read_records(root):
    records, rejected = [], []
    manifest = root / 'paper' / 'manifest.json'
    official = set()
    if manifest.is_file():
        for asset in read_json(manifest).get('assets', []):
            for item in asset.get('evaluations', []):
                path = item.get('path') if isinstance(item, dict) else item
                if path:
                    official.add((root / path).resolve())
    outputs = root / 'mmwave_densepose' / 'logs' / 'test'
    for path in sorted(outputs.rglob('metrics.json')) if outputs.exists() else []:
        run_path = path.with_name('run.json')
        if not run_path.is_file():
            continue
        try:
            run, metrics = read_json(run_path), normalized_metrics(read_json(path))
            if path.parent.resolve() in official or run.get('origin') == 'official' or path.with_name('paper-record.json').is_file():
                rejected.append({'record': relative(run_path, root), 'reason': 'imported_official_history_not_a_new_verified_run'})
                continue
            records.append({'record': relative(run_path, root), 'metrics_path': relative(path, root),
                            'run': run, 'metrics': metrics, 'directory': path.parent})
        except (OSError, ValueError, TypeError) as exc:
            rejected.append({'record': relative(run_path, root), 'reason': 'unreadable_metadata', 'error': type(exc).__name__})
    return records, rejected


def reference_tables(root):
    result = {}
    for n in range(1, 11):
        source = root / 'paper' / 'reference' / f'table_{n:02d}.csv'
        with source.open(encoding='utf-8-sig', newline='') as stream:
            raw = list(csv.reader(stream))
        if n <= 3:
            columns = [text(v) for v in raw[0]]
            rows = [[text(v) for v in row[:len(columns)]] + [''] * max(0, len(columns)-len(row))
                    for row in raw[1:] if row and row[0].strip()]
        elif n == 4:
            columns = ['Method', 'MLE (cm)', 'PA-MPJPE (cm)', 'PCK@15cm (%)', 'AUC@15cm (%)']
            rows = [[text(row[0]), *[number(v) for v in row[1:5]]] for row in raw[2:] if row and row[0].strip()]
        elif n == 5:
            columns = ['Method', *[f'J{i} (cm)' for i in range(19)], 'Average (cm)']
            rows = [[text(row[0]), *[number(v) for v in row[1:21]]] for row in raw[2:] if row and row[0].strip()]
        elif n == 6:
            columns = ['Method', 'Seed 42', 'Seed 50', 'Seed 66', 'Seed 69', 'Seed 87', 'Mean ± SD (cm)']
            rows = [[text(row[0]), *[number(v) for v in row[1:6]], text(row[6])] for row in raw[2:] if row and row[0].strip()]
        elif n == 7:
            columns = ['Setting', 'Method', 'Sequence-chunk MLE (cm)', 'Subject-independent MLE (cm)']
            rows, setting = [], ''
            for row in raw[2:]:
                if len(row) < 4 or not row[1].strip():
                    continue
                setting = text(row[0]) or setting
                rows.append([setting, text(row[1]), number(row[2]), number(row[3])])
        elif n == 8:
            columns = ['Model configuration', 'MLE (cm)']
            rows = [[text(row[0]), number(row[1])] for row in raw if len(row)>1 and row[0].strip().startswith('Baseline')]
        else:
            columns = ['Input features' if n == 9 else 'Input points', 'MLE (cm)']
            rows = [[text(row[0]), number(row[1])] for row in raw[1:] if len(row)>1 and row[0].strip()]
        # The reference is retained independently; nulls are real manuscript dashes.
        result[str(n)] = {'id': str(n), 'title': TITLES[n-1],
                         'reference': {'columns': columns, 'rows': rows},
                         'measured': {'columns': list(columns), 'rows': []}, 'provenance': [],
                         'reference_source': relative(source, root), 'reference_sha256': digest(source)}
    return result


def resolve_model(label):
    compact = ''.join(c for c in label.lower() if c.isalnum())
    aliases = [('pastnet', 'PASTNet'), ('pntposeat', 'PntPoseAT'), ('mmposefk', 'MMPoseFK'),
               ('dualpath', 'DualPathCNNBiLSTM'), ('globallocal', 'GlobalLocalNetwork'),
               ('posegraph', 'PoseGraphNet'), ('pointtransformer', 'PointTransformer'),
               ('pointmlp', 'PointMLP'), ('dgcnn', 'DGCNN')]
    return next((model for alias, model in aliases if alias in compact), None)


def build_snapshot(root):
    root = Path(root).resolve()
    tables = reference_tables(root)
    configs = {p.stem: read_json(p) for p in (root / 'paper' / 'configs').glob('*.json')}
    identities = {name: current_identity(root, cfg) for name, cfg in configs.items()}
    records, rejected = read_records(root)
    accepted = {}
    for name, config in configs.items():
        if config.get('model') == 'PCCCompletion':
            continue
        accepted[name] = []
        for record in records:
            try:
                reason = compatible_record(record['run'], record['metrics'], config, identities[name])
            except (TypeError, ValueError, KeyError):
                reason = 'invalid_configuration'
            if reason is None:
                accepted[name].append(record)
    included = sorted({r['record'] for group in accepted.values() for r in group})
    for record in records:
        if record['record'] not in included:
            try:
                matching = [name for name, cfg in configs.items() if fingerprint(record['run'].get('config', {})) == fingerprint(cfg)]
                name = matching[0] if matching else None
                reason = compatible_record(record['run'], record['metrics'], configs[name], identities[name]) if name else 'no_matching_paper_configuration'
            except (TypeError, ValueError, KeyError):
                reason = 'invalid_configuration'
            rejected.append({'record': record['record'], 'reason': reason})

    def put(table_id, row_index, column, experiment, metric='mle_cm', direction='min', joint=None):
        table = tables[str(table_id)]
        candidates = []
        for rec in accepted.get(experiment, []):
            value = rec['metrics'].get(metric)
            if joint is not None:
                if 'joint_metrics' not in rec:
                    path = rec['directory'] / 'per_joint_metrics.csv'
                    values = {}
                    try:
                        if path.is_file():
                            with path.open(encoding='utf-8-sig', newline='') as stream:
                                for entry in csv.DictReader(stream):
                                    if entry.get('joint_index', '').strip().lower() == 'mean':
                                        continue
                                    index = int(entry['joint_index'])
                                    if index in values or not 0 <= index < 19:
                                        raise ValueError('Duplicate or invalid joint index')
                                    numeric = number(entry.get('mle_m'))
                                    values[index] = None if numeric is None or numeric < 0 else numeric * 100
                    except (OSError, ValueError, KeyError):
                        values = {}
                    rec['joint_metrics'] = values
                value = rec['joint_metrics'].get(joint)
            if finite(value):
                candidates.append((value, rec))
        if not candidates:
            return None
        value, record = (min if direction == 'min' else max)(candidates, key=lambda pair: (pair[0], pair[1]['record']))
        table['measured']['rows'][row_index][column] = value
        run = record['run']
        table['provenance'].append({'row': row_index, 'column': column, 'value': value,
                                    'metric': f'joint_{joint}_mle_cm' if joint is not None else metric,
                                    'direction': direction, 'experiment': experiment,
                                    'record': record['record'], 'metrics': record['metrics_path'],
                                    'checkpoint': run.get('checkpoint'), 'checkpoint_sha256': run.get('checkpoint_sha256'),
                                    'dataset_manifest_sha256': run.get('dataset_manifest_sha256')})
        return value

    for n in range(1, 11):
        table = tables[str(n)]
        label_columns = 2 if n == 7 else 1
        table['measured']['rows'] = [row[:label_columns] + [None] * (len(table['reference']['columns'])-label_columns)
                                      for row in table['reference']['rows']]
        if n == 4:
            for i, experiment in enumerate(MAIN):
                for column, (metric, direction) in enumerate(METRICS, 1):
                    put(n, i, column, experiment, metric, direction)
        elif n == 5:
            for i, experiment in enumerate(MAIN):
                for joint in range(19):
                    put(n, i, joint + 1, experiment, joint=joint)
                put(n, i, 20, experiment)
        elif n == 6:
            for i, row in enumerate(table['reference']['rows']):
                model = resolve_model(row[0]); prefix = 'pastnet' if model == 'PASTNet' else 'pnt-pose-at'
                values = []
                for column, seed in enumerate((42, 50, 66, 69, 87), 1):
                    experiment = MODEL_EXPERIMENT[model] if seed == 42 else f'{prefix}-seed{seed}'
                    values.append(put(n, i, column, experiment))
                if all(v is not None for v in values):
                    table['measured']['rows'][i][6] = f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}'
                    table['provenance'].append({'row': i, 'column': 6, 'metric': 'five_seed_mean_sample_std', 'seeds': [42, 50, 66, 69, 87], 'value': table['measured']['rows'][i][6], 'records': [p['record'] for p in table['provenance'] if p['row'] == i and 1 <= p['column'] <= 5]})
        elif n == 7:
            for i, row in enumerate(table['reference']['rows']):
                main = MODEL_EXPERIMENT[resolve_model(row[1])]
                put(n, i, 2, main)
                put(n, i, 3, 'subject-' + main.removeprefix('main-'))
        elif n in (8, 9, 10):
            names = {8: ['ablation-pointnet', 'ablation-spatial', 'main-pastnet'],
                     9: ['ablation-input-3d', 'main-pastnet'],
                     10: [*[f'points-{points}' for points in (25, 50, 75, 100, 125, 150)], 'main-pastnet']}[n]
            for i, name in enumerate(names):
                put(n, i, 1, name)
    return {'tables': tables, 'snapshot': {'schema_version': 1,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'selection': 'Per-cell best compatible complete test: minimize MLE/PA-MPJPE, maximize PCK/AUC.',
            'missing_value': None, 'records_scanned': len(records), 'records_included': included,
            'records_rejected': rejected,
            'identities': {name: {k: v for k, v in identity.items() if k != 'metadata'}
                           for name, identity in identities.items() if identity}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    snapshot = build_snapshot(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output / 'tables.json').write_text(encoded + '\n', encoding='utf-8')
    (args.output / 'tables.js').write_text('globalThis.PAPER_TABLE_DATA = ' + encoded.replace('</', '<\\/') + ';\n', encoding='utf-8')
    print(json.dumps({'tables': len(snapshot['tables']), 'records_scanned': snapshot['snapshot']['records_scanned'],
                      'records_included': len(snapshot['snapshot']['records_included'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
