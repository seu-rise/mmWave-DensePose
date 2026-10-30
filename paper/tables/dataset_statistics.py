"""Audit installed numeric data while preserving manuscript statistics separately."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmwave_densepose.data.cache import training_root


def audit(data_root, output_dir):
    source_root = data_root
    data_root = training_root(data_root, 'chunk_s42')
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('partial'):
        raise ValueError('Dataset statistics require a complete installation, not a partial subset')
    sequences = manifest['sequences']
    measurements = dict(sequences=len(sequences), anonymous_subject_ids=len({entry['subject_id'] for entry in sequences.values()}),
                        stored_frames=0, stored_radar_points=0, finite_xyz_radar_points=0,
                        frames_over_50_finite_xyz_points=0, finite_xyz_points_in_valid_frames=0,
                        frames_with_finite_19_joint_labels=0)
    for count, entry in enumerate(sequences.values(), 1):
        with np.load(data_root / entry['file'], allow_pickle=False) as arrays:
            points, offsets, joints = (arrays[key] for key in ('points', 'offsets', 'joints'))
            if offsets.shape != (entry['frames'] + 1,) or joints.shape != (entry['frames'], 19, 3):
                raise ValueError(f"Invalid numeric sequence: {entry['file']}")
            if offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
                raise ValueError(f"Invalid radar offsets: {entry['file']}")
            finite = np.isfinite(points[:, :3]).all(axis=1)
            prefix = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
            valid_counts = np.diff(prefix[offsets])
            measurements['stored_frames'] += len(joints)
            measurements['stored_radar_points'] += len(points)
            measurements['finite_xyz_radar_points'] += int(finite.sum())
            measurements['frames_over_50_finite_xyz_points'] += int((valid_counts > 50).sum())
            measurements['finite_xyz_points_in_valid_frames'] += int(valid_counts[valid_counts > 50].sum())
            measurements['frames_with_finite_19_joint_labels'] += int(np.isfinite(joints).all(axis=(1, 2)).sum())
        if count % 20 == 0:
            print(f'Audited {count}/{len(sequences)} sequences', flush=True)
    frames = measurements['stored_frames']
    measurements['mean_stored_frames_per_sequence'] = frames / len(sequences)
    measurements['mean_finite_xyz_points_per_stored_frame'] = measurements['finite_xyz_radar_points'] / frames
    valid_frames = measurements['frames_over_50_finite_xyz_points']
    measurements['mean_valid_frames_per_sequence'] = valid_frames / len(sequences)
    measurements['mean_finite_xyz_points_per_valid_frame'] = (
        measurements['finite_xyz_points_in_valid_frames'] / valid_frames if valid_frames else None)
    split_counts = {}
    for name in manifest['splits']:
        metadata = json.loads((data_root / 'splits' / (name + '.json')).read_text(encoding='utf-8'))
        with np.load(data_root / 'splits' / (name + '.npz'), allow_pickle=False) as arrays:
            actual = {split: len(arrays[split]) for split in ('train', 'val', 'test')}
        if actual != metadata['split_num_samples']:
            raise ValueError(f'Frozen split count mismatch: {name}')
        split_counts[name] = dict(seq_len=metadata['seq_len'], split_strategy=metadata['split_strategy'], samples=actual)
    with (ROOT / 'paper/reference/table_02.csv').open(encoding='utf-8') as stream:
        paper_rows = list(csv.DictReader(stream))
    result = dict(status='audited', manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  cache_measurements=measurements, frozen_splits=split_counts,
                  paper_reference=dict(source='paper/reference/table_02.csv', rows=paper_rows),
                  interpretation=[
                      'Stored cache frames and anonymous subject IDs describe the installed numeric cache.',
                      'Valid-point frames apply the source strict greater-than-50 finite XYZ points criterion.',
                      'The manuscript frame count can be compared against this observed valid-point population.',
                      'Original timestamps remain in the full PKLs; this numeric-cache auditor does not re-evaluate timestamp continuity.',
                      'No measured cache count is substituted for a differently defined manuscript statistic.',
                      'Sequence contents are counted as stored; aliases and any duplicated content are not silently deduplicated.'])
    output_dir.mkdir(parents=True, exist_ok=True)
    result['source_dataset_manifest_sha256'] = hashlib.sha256((source_root / 'manifest.json').read_bytes()).hexdigest()
    (output_dir / 'table_02_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'table_02_cache_measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['measurement', 'value', 'population'])
        writer.writerows([key, value, 'installed_numeric_cache'] for key, value in measurements.items())
    print(json.dumps(measurements, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/mmwave_densepose')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.local/runtime/dataset-statistics')
    args = parser.parse_args()
    audit(args.data_root, args.output_dir)


if __name__ == '__main__':
    main()
