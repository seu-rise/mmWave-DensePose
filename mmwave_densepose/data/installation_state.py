"""Cheap readiness guard for model tasks against a complete release inventory.

This guard never scans PKL content or hashes source sequences. Dataset browsing
does not use it; partially installed releases remain available to the viewer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


class DatasetIncompleteError(ValueError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.issue = {
            'name': 'data_root', 'group': 'data', 'severity': 'error',
            'code': 'dataset_incomplete', 'message': message,
            'suggestion': 'Open Dataset and continue the download. Wait for all packages to be installed and verified before starting training, validation, testing or prediction. Available sequences can still be browsed.',
        }
        if status:
            self.issue['actual'] = status
            self.issue['expected'] = 'Complete installed and verified dataset release'


def _read_mapping(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected an object')
    return value


def _present(root, name, size, *, raw=False):
    if not isinstance(name, str) or not name or '\\' in name:
        return False
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or any(part in ('', '.', '..') for part in relative.parts):
        return False
    path = root.joinpath(*relative.parts)
    if raw and (relative.parts[0] != 'raw' or path.suffix.lower() not in ('.pkl', '.pickle')):
        return False
    try:
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return False
        return (path.resolve().is_relative_to(root) and path.is_file()
                and type(size) is int and size >= 0 and path.stat().st_size == size)
    except OSError:
        return False


def ensure_complete_dataset(data_root, manifest=None):
    """Reject incomplete full releases; leave intentional numeric fixtures alone.

    A persisted in-progress marker remains blocking even if every file has been
    extracted: only the installer/verification workflow may declare completion.
    Manually installed full datasets with no marker retain inventory-based use.
    """
    root = Path(data_root).expanduser().resolve()
    marker_path = root / 'cache' / 'installation-state.json'
    marker = None
    if marker_path.exists() or marker_path.is_symlink():
        try:
            if marker_path.is_symlink() or marker_path.parent.is_symlink():
                raise ValueError('Installation state must not be linked')
            marker = _read_mapping(marker_path)
        except (OSError, ValueError, TypeError):
            raise DatasetIncompleteError('Dataset installation state is invalid; continue the download or verify the complete release.') from None
    manifest_path = root / 'manifest.json'
    if manifest is None:
        try:
            manifest = _read_mapping(manifest_path)
        except (OSError, ValueError, TypeError):
            raw_directory = root / 'raw'
            legacy_sequences = raw_directory.is_dir() and any(
                item.is_file() and item.suffix.lower() in ('.pkl', '.pickle')
                for item in raw_directory.iterdir())
            if marker is not None or legacy_sequences:
                raise DatasetIncompleteError('Dataset download is incomplete: verified dataset metadata has not been installed yet.') from None
            return None
    if manifest.get('dataset_kind') != 'full_multimodal' and marker is None:
        return None
    sequences = manifest.get('sequences')
    files = manifest.get('files', {})
    if not isinstance(sequences, dict) or not sequences or not isinstance(files, dict):
        raise DatasetIncompleteError('Dataset download is incomplete: the full release inventory is unavailable.')
    installed_sequences = sum(isinstance(item, dict) and _present(root, item.get('file'), item.get('size_bytes'), raw=True)
                              for item in sequences.values())
    inventory = dict(files)
    for item in sequences.values():
        if isinstance(item, dict) and isinstance(item.get('file'), str):
            inventory[item['file']] = item
    manifest_present = manifest_path.is_file() and not manifest_path.is_symlink()
    installed_files = int(manifest_present) + sum(isinstance(item, dict) and _present(root, name, item.get('size_bytes'))
                                                  for name, item in inventory.items())
    status = {'installed_sequences': installed_sequences, 'total_sequences': len(sequences),
              'installed_files': installed_files, 'total_files': len(inventory) + 1}
    if marker is not None:
        try:
            matches = (isinstance(marker.get('dataset_manifest_sha256'), str)
                       and hashlib.sha256(manifest_path.read_bytes()).hexdigest() == marker['dataset_manifest_sha256'])
        except OSError:
            matches = False
        if not matches:
            raise DatasetIncompleteError('Dataset installation metadata does not match its recorded release.', status)
        if marker.get('complete') is not True:
            raise DatasetIncompleteError('Dataset download is incomplete or final verification has not finished.', status)
    if (manifest.get('partial') or installed_sequences != len(sequences)
            or installed_files != len(inventory) + 1):
        raise DatasetIncompleteError(f'Dataset download is incomplete: {installed_sequences} of {len(sequences)} sequences are installed.', status)
    return status
