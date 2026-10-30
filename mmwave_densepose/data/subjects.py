"""Canonical subject identities for the prepared mmWave-DensePose dataset."""
from __future__ import annotations

import copy
import json
from pathlib import Path

CANONICAL_MAPPING = Path(__file__).resolve().parents[2] / 'dataset/subject-mapping.json'


def load_subject_mapping(path, canonical_path=CANONICAL_MAPPING):
    document = json.loads(Path(path).read_text(encoding='utf8'))
    mapping = dict(document['sequences'])
    canonical = json.loads(Path(canonical_path).read_text(encoding='utf8'))['sequences']
    if not isinstance(canonical, dict) or not all(isinstance(key,str) and isinstance(value,str) and value for key,value in canonical.items()):
        raise ValueError('Canonical subject mapping is invalid')
    for filename in mapping:
        if filename in canonical:
            mapping[filename] = canonical[filename]
    return mapping


def correct_index_subjects(index, subjects):
    """Correct identity metadata while retaining every original sample and frame."""
    signature = index.get('dataset_signature', [])
    corrected = []
    changes = []
    for row in signature:
        replacement = subjects.get(row[2]) if len(row)>3 else None
        if replacement and replacement != row[3]:
            updated = list(row)
            updated[3] = replacement
            corrected.append(tuple(updated) if isinstance(row,tuple) else updated)
            changes.append((row[0],row[3],replacement))
        else:
            corrected.append(row)
    if not changes:
        return index, False
    result = copy.deepcopy(index)
    result['dataset_signature'] = tuple(corrected) if isinstance(signature,tuple) else corrected
    labels = {row[0]:row[3] for row in corrected if len(row)>3}
    aliases = [row[0] for row in corrected]
    for split, existing in result.get('split_subject_ids',{}).items():
        members = {row[0] if isinstance(row[0],str) else aliases[int(row[0])]
                   for row in index.get('samples',{}).get(split,[])}
        used = {labels[alias] for alias in members if alias in labels}
        names = set(existing)
        for alias,old,new in changes:
            if alias in members:
                names.add(new)
                if old not in used:
                    names.discard(old)
        result['split_subject_ids'][split] = sorted(names)
        stats = result.get('stats',{}).get('split_num_subjects')
        if isinstance(stats,dict) and split in stats:
            stats[split] = len(names)
    return result, True
