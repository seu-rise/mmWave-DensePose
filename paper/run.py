"""Run the experiments corresponding to a manuscript table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import statistics
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmwave_densepose.engine.artifacts import artifact_roots, resolve_run_paths
MAIN = ['main-dgcnn', 'main-point-transformer', 'main-pointmlp', 'main-mm-pose-fk',
        'main-dual-path', 'main-pnt-pose-at', 'main-pose-graph', 'main-global-local', 'main-pastnet']
SUBJECT = ['subject-' + name.removeprefix('main-') for name in MAIN]
TABLES = {
    4: MAIN,
    5: MAIN,
    6: ['main-pastnet', *[f'pastnet-seed{s}' for s in (50, 66, 69, 87)],
        'main-pnt-pose-at', *[f'pnt-pose-at-seed{s}' for s in (50, 66, 69, 87)]],
    7: MAIN + SUBJECT,
    8: ['ablation-pointnet', 'ablation-spatial', 'main-pastnet'],
    9: ['ablation-input-3d', 'main-pastnet'],
    10: [*[f'points-{n}' for n in (25, 50, 75, 100, 125, 150)], 'main-pastnet'],
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--table', type=int, choices=range(1, 11))
    group.add_argument('--all', action='store_true')
    parser.add_argument('--action', choices=['evaluate', 'validate', 'train'], default='evaluate')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir', type=Path, help='Aggregate report directory; experiment logs and weights use the managed user layout')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit-batches', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--completion-checkpoint', help='Explicit PCC weights for main GlobalLocalNetwork training')
    parser.add_argument('--subject-completion-checkpoint', help='Explicit PCC weights for subject GlobalLocalNetwork training')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir = (args.output_dir or artifact_roots(ROOT)['runtime'] / 'paper-reproduction'
                       / ('report_' + uuid.uuid4().hex[:10])).resolve()
    if args.output_dir.is_relative_to(ROOT / 'paper'):
        parser.error('Reproduction reports must not overwrite prepared official paper materials')
    tables = list(range(1, 11)) if args.all else [args.table]
    tasks = list(dict.fromkeys(exp for table in tables for exp in TABLES.get(table, [])))
    completion_weights = {'main-global-local': args.completion_checkpoint,
                          'subject-global-local': args.subject_completion_checkpoint}
    if args.action != 'train' and any(completion_weights.values()):
        parser.error('PCC selection is for training; evaluation uses the complete pose checkpoint')
    if args.action == 'train':
        for experiment, path in completion_weights.items():
            flag = '--completion-checkpoint' if experiment == 'main-global-local' else '--subject-completion-checkpoint'
            if path and experiment not in tasks:
                parser.error(f'{flag} does not apply to this selection. No training was started.')
            if experiment in tasks and not args.dry_run:
                if not path or not Path(path).is_file():
                    parser.error(f'{experiment} requires an existing {flag} PATH. No training was started.')
                # Validate every dependency before any table audit or training starts.
                from mmwave_densepose.engine.runner import load_weights
                from mmwave_densepose.models import build_model
                try:
                    dependency = build_model('PCCCompletion', {'num_points': 80, 'num_out_points': 196})
                    load_weights(dependency, path, 'PCCCompletion')
                    del dependency
                except (ValueError, RuntimeError, OSError) as exc:
                    parser.error(f'Invalid {flag}: {exc}. No training was started.')
    report = dict(action=args.action, requested_tables=tables, dry_run=args.dry_run,
                  limited_run=args.limit_batches is not None, tables={}, experiments=[])
    for table in tables:
        report['tables'][str(table)] = {'reference': f'paper/reference/table_{table:02d}.csv',
            'status': 'descriptive_reference' if table in (1, 3) else 'pending',
            'experiments': TABLES.get(table, [])}
    failed = False
    if 2 in tables:
        command = [sys.executable, str(ROOT / 'paper/tables/dataset_statistics.py'),
                   '--data-root', str(Path(args.data_root).resolve()) if args.data_root else str(ROOT / 'data/mmwave_densepose'),
                   '--output-dir', str(args.output_dir.resolve())]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            report['tables']['2']['status'] = 'planned'
        else:
            result = subprocess.run(command, cwd=ROOT, check=False)
            failed = result.returncode != 0
            report['tables']['2'].update(status='failed' if failed else 'audited', returncode=result.returncode)
            if failed and not args.continue_on_error:
                write_json(args.output_dir / f'{args.action}-status.json', report)
                return 1
    for experiment in tasks:
        config = ROOT / 'paper/configs' / f'{experiment}.json'
        effective = json.loads(config.read_text(encoding='utf-8'))
        if args.batch_size is not None:
            effective['batch_size'] = args.batch_size
        selected = effective.get('checkpoint') if args.action != 'train' else None
        output = resolve_run_paths(args.action, effective, checkpoint=selected, root=ROOT)['log_dir']
        command = [sys.executable, str(ROOT / 'scripts' / f'{args.action}.py'),
                   '--config', str(config), '--output-dir', str(output), '--device', args.device]
        if args.data_root:
            command += ['--data-root', str(Path(args.data_root).resolve())]
        if args.limit_batches is not None:
            command += ['--limit-batches', str(args.limit_batches)]
        if args.batch_size is not None:
            command += ['--batch-size', str(args.batch_size)]
        if args.action == 'train' and completion_weights.get(experiment):
            command += ['--completion-checkpoint', str(Path(completion_weights[experiment]).resolve())]
        print(shlex.join(command), flush=True)
        record = dict(id=experiment, config=config.relative_to(ROOT).as_posix(), output_dir=output.relative_to(ROOT).as_posix(),
                      status='planned' if args.dry_run else 'running')
        if not args.dry_run:
            start = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, check=False)
            record.update(returncode=result.returncode, seconds=round(time.monotonic() - start, 3),
                          status=('smoke_completed' if args.limit_batches is not None else 'completed')
                          if result.returncode == 0 else 'failed')
            metrics_file = output / 'metrics.json'
            if result.returncode == 0 and args.action != 'train':
                if metrics_file.is_file():
                    record['metrics'] = json.loads(metrics_file.read_text(encoding='utf-8'))
                else:
                    record.update(status='failed', error='Evaluation succeeded without metrics.json')
            failed = failed or record['status'] == 'failed'
        report['experiments'].append(record)
        if failed and not args.continue_on_error:
            break
    completed = {record['id'] for record in report['experiments']
                 if record['status'] in ('completed', 'smoke_completed')}
    for table, record in report['tables'].items():
        expected = record['experiments']
        if expected:
            record['status'] = 'planned' if args.dry_run else (
                ('smoke_checked' if args.limit_batches is not None else 'completed')
                if set(expected).issubset(completed) else 'incomplete')
    if args.dry_run:
        print(json.dumps(report, indent=2))
    else:
        write_json(args.output_dir / f'{args.action}-status.json', report)
        if completed:
            target = args.output_dir / f'{args.action}-metrics.csv'
            metric_rows = []
            for item in report['experiments']:
                metrics = item.get('metrics', {})
                scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
                if scalar:
                    metric_rows.append(dict(experiment=item['id'], **scalar))
            if metric_rows:
                fields = list(dict.fromkeys(k for row in metric_rows for k in row))
                with target.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(metric_rows)
                for table in tables:
                    selected = [row for row in metric_rows if row['experiment'] in TABLES.get(table, [])]
                    if selected:
                        table_path = args.output_dir / f'table_{table:02d}_{args.action}.csv'
                        with table_path.open('w', newline='', encoding='utf-8') as stream:
                            writer = csv.DictWriter(stream, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(selected)
                if 6 in tables:
                    groups = {'PASTNet': TABLES[6][:5], 'PntPoseAT': TABLES[6][5:]}
                    summary = {}
                    for name, experiments in groups.items():
                        values = [row['mle_cm'] for row in metric_rows
                                  if row['experiment'] in experiments and 'mle_cm' in row]
                        summary[name] = dict(completed_runs=len(values), expected_runs=5)
                        if len(values) == 5:
                            summary[name].update(mean_mle_cm=statistics.mean(values),
                                                 sample_std_mle_cm=statistics.stdev(values))
                    write_json(args.output_dir / 'table_06_statistics.json',
                               dict(action=args.action, limited_run=args.limit_batches is not None,
                                    models=summary))
                if 5 in tables:
                    joints = []
                    for experiment in TABLES[5]:
                        matched = next((item for item in report['experiments'] if item['id'] == experiment), None)
                        per_joint = ROOT / matched['output_dir'] / 'per_joint_metrics.csv' if matched else Path('missing')
                        if experiment in completed and per_joint.is_file():
                            with per_joint.open(encoding='utf-8') as stream:
                                joints.extend(dict(experiment=experiment, **row) for row in csv.DictReader(stream))
                    if joints:
                        with (args.output_dir / 'table_05_per_joint.csv').open('w', newline='', encoding='utf-8') as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(joints[0]))
                            writer.writeheader()
                            writer.writerows(joints)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
