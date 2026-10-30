"""Index for test runs and recoverable deletion of local evaluation artefacts.

The web server must pass fixed output/data roots. Values from run metadata are
never treated as paths supplied by the browser.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
import threading
import uuid

import numpy as np

from .legacy_results import read_paper_result
from mmwave_densepose.workflows.catalog import official_asset_for_log


METRIC_KEYS = ("mle_cm", "pa_mpjpe_cm", "pck_percent", "auc_percent",
               "samples", "joints", "threshold_m", "cd_m", "cd_cm")
CONFIG_KEYS = ("split_strategy", "split_manifest", "seed", "partition", "split_chunk_size",
               "max_timestamp_gap_ms", "min_valid_points", "num_points", "input_dim",
               "seq_len", "num_joints", "ablation")
SAMPLE_ID_RE = re.compile(r"^(?P<sequence>[^:]+):(?P<start>\d+):(?P<end>\d+)$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _csv_number(value):
    """Parse CSV cells without turning blanks or invalid numbers into zero."""
    if value is None or not str(value).strip():
        return None
    try:
        return _number(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_json(value):
    """Serialize numeric sample arrays while preserving missing coordinates."""
    if isinstance(value,np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value,np.generic):
        return _safe_json(value.item())
    if isinstance(value,(list,tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value,dict):
        return {str(key):_safe_json(item) for key,item in value.items()}
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def parse_sample_id(value):
    match = SAMPLE_ID_RE.fullmatch(str(value or ""))
    if not match:
        return None
    start, end = int(match["start"]), int(match["end"])
    if end <= start:
        return None
    return {"sequence_id": match["sequence"], "start": start, "end": end,
            "target_frame": end - 1, "sample_id": str(value)}


class ResultsIndex:
    """Indexes immutable run snapshots below operator-selected local roots."""

    def __init__(self, outputs_root, *, reproduce_results_root=None, data_root=None,
                 trusted_pickle=False, training_logs_root=None, repository_root=None, unified_logs=False):
        self.unified_logs = bool(unified_logs)
        self.training_root = Path(training_logs_root).expanduser().resolve() if training_logs_root else None
        self.outputs_root = Path(outputs_root).expanduser().resolve()
        self.reproduce_root = (Path(reproduce_results_root).expanduser().resolve()
                               if reproduce_results_root else None)
        self.repository_root = Path(repository_root).resolve() if repository_root else self.outputs_root.parent
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self.trusted_pickle = bool(trusted_pickle)
        self._records = None
        self._records_at = 0.0
        self._lock = threading.RLock()
        self._visual_cache_key = None
        self._visual_cache = None
        self._visual_cache_validated = False
        self._visual_index_payload = None
        self._visual_index = None
        self._visual_requested_records = set()

    @property
    def roots(self):
        return tuple(root for root in (self.outputs_root, self.reproduce_root) if root is not None)

    def _run_paths(self):
        """Find small run manifests without following symlinks into untrusted paths."""
        scan_roots = tuple(dict.fromkeys((*self.roots, *((self.training_root,) if self.unified_logs and self.training_root else ()))))
        for root in scan_roots:
            if not root.is_dir():
                continue
            def walk_error(_error):
                return None
            for current, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
                directory = Path(current)
                if directory.is_symlink() or not _inside(directory.resolve(), root):
                    dirs[:] = []
                    continue
                dirs[:] = [name for name in dirs if not name.startswith(".") and
                           not (directory / name).is_symlink() and
                           _inside((directory / name).resolve(), root)]
                if self.unified_logs:
                    phase = 'train' if root == self.training_root else 'test'
                    official_name = 'paper-training.json' if phase == 'train' else 'paper-record.json'
                    metadata = _read_json(directory / official_name) if official_name in files else None
                    authentic = official_asset_for_log(directory, self.repository_root, phase=phase, metadata=metadata)
                    name = official_name if authentic else 'run.json' if 'run.json' in files else None
                else:
                    name = "run.json" if "run.json" in files else "paper-record.json" if root == self.reproduce_root and "paper-record.json" in files else None
                if name:
                    manifest_path = directory / name
                    if not manifest_path.is_symlink() and _inside(manifest_path.resolve(), root):
                        yield root, manifest_path

    def _record_source(self, root, info, directory=None):
        if self.unified_logs:
            phase = 'train' if root == self.training_root else 'test'
            asset = (official_asset_for_log(directory, self.repository_root, phase=phase, metadata=info)
                     if directory is not None else None)
            return "official" if asset else "user"
        return "user" if root == self.outputs_root else "official"

    def _record_id(self, root, relative, source):
        if self.unified_logs:
            phase = "training-" if root == self.training_root else ""
            return f"logs-{phase}{source}:{relative}"
        return f"outputs:{relative}" if root == self.outputs_root else f"reproduce-results:{relative}"

    def _load_records(self, refresh=False):
        with self._lock:
            if self._records is not None and not refresh and time.monotonic() - self._records_at < 2.0:
                return self._records
            return self._scan_records()

    def invalidate(self):
        """Discover explicitly installed predictions without retaining old missing-file probes."""
        with self._lock:
            self._records = None
            self._records_at = 0.0
            self._visual_cache = None
            self._visual_cache_key = None
            self._visual_cache_validated = False
            self._visual_index_payload = self._visual_index = None

    def _scan_records(self):
        runs, train_runs = [], []
        seen = set()
        for root, manifest_path in self._run_paths():
            relative = _relative(manifest_path.parent, root)
            identity = (str(root), relative)
            if identity in seen:
                continue
            seen.add(identity)
            info = _read_json(manifest_path)
            if info is None:
                continue
            if manifest_path.name == 'paper-training.json':
                asset = official_asset_for_log(manifest_path.parent, self.repository_root,
                                               phase='train', metadata=info)
                if asset:
                    train_runs.append({'record_id': self._record_id(root, relative, 'official'),
                        'source': 'official', 'model': asset['model'], 'action': 'train',
                        'weight_id': info.get('weight_id') or 'paper:' + asset['id'],
                        'checkpoint': asset.get('path'), 'checkpoint_sha256': asset.get('sha256'),
                        'paper_tables': asset.get('paper_tables', info.get('paper_tables', [])),
                        'paper_dependency_tables': asset.get('paper_dependency_tables', info.get('paper_dependency_tables', [])),
                        'training_run_id': info.get('training_run_id') or info.get('run_id') or manifest_path.parent.name,
                        'config': info.get('config', {}), 'mtime': manifest_path.stat().st_mtime})
                continue
            if manifest_path.name == "paper-record.json":
                record = self._paper_record(root, manifest_path, info)
                if record is not None:
                    runs.append(record)
                continue
            action = str(info.get("action", "")).lower()
            if self.unified_logs and ((root == self.training_root and action != "train") or (root == self.outputs_root and action not in ("evaluate", "test"))):
                continue
            raw_config=info.get("config")
            config=raw_config if isinstance(raw_config,dict) else {}
            model = str(info.get("model") or config.get("model") or "").strip()
            if not model:
                continue
            source = self._record_source(root, info, manifest_path.parent)
            record_id = self._record_id(root, relative, source)
            metrics_path = manifest_path.parent / "metrics.json"
            metrics = (_read_json(metrics_path) or {} if not metrics_path.is_symlink() and
                       _inside(metrics_path.resolve(), root) else {})
            checkpoint = info.get("checkpoint") or (config.get("checkpoint") if action != "train" else None)
            checkpoint_hash = info.get("checkpoint_sha256")
            if action=="train":
                checkpoint=None
                checkpoint_hash=info.get("checkpoint_sha256")
            if not checkpoint_hash and isinstance(checkpoint, str):
                candidate = Path(checkpoint)
                if not candidate.is_absolute():
                    candidate = self.repository_root / candidate
                try:
                    resolved_checkpoint=candidate.resolve(strict=True)
                except OSError:
                    resolved_checkpoint=None
                if (resolved_checkpoint and any(_inside(resolved_checkpoint, allowed) for allowed in
                                                (self.repository_root, *self.roots)) and
                        candidate.is_file() and not candidate.is_symlink()):
                    try:
                        checkpoint_hash = self._sha256(candidate)
                    except OSError:
                        checkpoint_hash = None
            sample_count = metrics.get("samples")
            split_meta = info.get("split_metadata") if isinstance(info.get("split_metadata"), dict) else {}
            split = info.get("split") or ("test" if action in ("evaluate", "infer", "test") else "val" if action == "validate" else "train")
            expected_counts = split_meta.get("split_num_samples", {})
            expected_count = expected_counts.get(split) if isinstance(expected_counts, dict) else None
            limited = info.get("limit_batches") is not None
            partial = bool(info.get("partial") or info.get("allow_partial") or metrics.get("partial"))
            complete = (info.get("evaluation_complete") is True or
                        (action in ("evaluate", "test") and not limited and not partial and
                         isinstance(sample_count, int) and sample_count > 0 and
                         isinstance(expected_count, int) and sample_count == expected_count))
            state = "complete" if complete else "incomplete"
            mtime = manifest_path.stat().st_mtime
            if action == "train":
                summary_path = manifest_path.parent / "training_summary.json"
                summary = (_read_json(summary_path) or {} if not summary_path.is_symlink() and
                           _inside(summary_path.resolve(), root) else {})
                best_path = summary.get("best_checkpoint")
                if best_path:
                    checkpoint = best_path
                    resolved = Path(best_path)
                    candidates = [resolved] if resolved.is_absolute() else [manifest_path.parent / resolved,
                                                                           self.repository_root / resolved]
                    resolved = next((candidate for candidate in candidates if candidate.is_file() and
                                     not candidate.is_symlink() and any(_inside(candidate.resolve(), allowed)
                                     for allowed in (self.repository_root, *self.roots))), None)
                    if resolved:
                        checkpoint = resolved
                        try:
                            checkpoint_hash = self._sha256(resolved)
                        except OSError:
                            checkpoint_hash = None
                safe_checkpoint = self._relative_checkpoint(checkpoint, manifest_path.parent) if checkpoint else None
                train_runs.append({"record_id": record_id, "source": source, "model": model, "action": action,
                                   "checkpoint": safe_checkpoint,
                                   "checkpoint_sha256": checkpoint_hash,
                                   "training_run_id": info.get("run_id") or manifest_path.parent.name,
                                   "config": config, "mtime": mtime})
                continue
            if action not in ("evaluate", "test"):
                continue
            pkl_path = self._prediction_path(manifest_path.parent, info)
            pkl_present = pkl_path is not None and pkl_path.is_file() and not pkl_path.is_symlink()
            relative_pkl = f"{record_id}/{_relative(pkl_path, manifest_path.parent)}" if pkl_present else None
            pkl_valid = None
            safe_checkpoint = self._relative_checkpoint(checkpoint, manifest_path.parent) if checkpoint else None
            training = self._matching_training(model, checkpoint_hash, safe_checkpoint, train_runs, root, source=source)
            # Training manifests may occur after test manifests in lexical walk order.
            if training is None:
                training = {"training_run_id": (f"pretrained-{checkpoint_hash[:12]}" if checkpoint_hash else
                                  f"checkpoint-{Path(str(checkpoint or 'unknown')).stem}"),
                            "checkpoint": safe_checkpoint,
                            "checkpoint_sha256": checkpoint_hash}
            expected_hash = info.get("dataset_manifest_sha256")
            index_hash = self._split_index_hash(config.get("split_manifest"), root, config=config, recorded=info)
            metric_values = {key: _number(metrics.get(key)) for key in METRIC_KEYS}
            per_joint = self._per_joint(manifest_path.parent, root)
            npz_manifest = _read_json(manifest_path.parent / "predictions" / "manifest.json")
            prediction_kind = npz_manifest.get("format") if npz_manifest else None
            runs.append({"record_id": record_id, "source": source, "test_id": info.get("test_id") or manifest_path.parent.name,
                         "training_run_id": info.get("training_run_id") or training["training_run_id"], "weight_id": info.get("weight_id") or checkpoint_hash or str(checkpoint or training["training_run_id"]),
                         "checkpoint": safe_checkpoint,
                         "checkpoint_sha256": checkpoint_hash, "model": model, "action": action,
                         "split": split, "split_manifest": info.get("split_manifest") or config.get("split_manifest"),
                         "split_strategy": info.get("split_strategy") or config.get("split_strategy"),
                         "partition_cache_id": info.get("partition_cache_id"),
                         "seed": info.get("seed", config.get("seed")), "config": config,
                         "dataset_manifest_sha256": expected_hash, "split_index_sha256": index_hash,
                         "partial": partial, "limited": limited, "complete": complete, "status": state,
                         "samples": sample_count if isinstance(sample_count, int) else None,
                         "expected_samples": expected_count if isinstance(expected_count, int) else None,
                         "joints": metric_values["joints"], "threshold_m": metric_values["threshold_m"],
                         "metrics": metric_values, "per_joint": per_joint,
                         "prediction_npz": {"available": bool(npz_manifest and prediction_kind),
                                            "format": prediction_kind, "manifest": f"{record_id}/predictions/manifest.json" if npz_manifest else None},
                         "prediction_pkl": {"present": bool(pkl_present), "valid": bool(pkl_valid),
                                            "available": bool(pkl_valid), "path": relative_pkl,
                                            "trusted": self.trusted_pickle},
                         "_pkl_relative":_relative(pkl_path,manifest_path.parent) if pkl_present else None,
                         "mtime": mtime})
        train_runs.sort(key=lambda row: (row["model"].casefold(), row["training_run_id"]))
        training_map = {(row["source"], row["model"], row.get("checkpoint_sha256")): row for row in train_runs if row.get("checkpoint_sha256")}
        for row in runs:
            if row.get("checkpoint_sha256") and (row["source"], row["model"], row["checkpoint_sha256"]) in training_map:
                row["training_run_id"] = training_map[(row["source"], row["model"], row["checkpoint_sha256"])]["training_run_id"]
        runs.sort(key=lambda row: (-row["mtime"], row["record_id"]))
        self._records = {"tests": runs, "training": train_runs}
        self._records_at = time.monotonic()
        return self._records

    def _paper_record(self, root, path, metadata):
        if metadata.get("schema_version") != 1 or not isinstance(metadata.get("model"), str):
            return None
        if not metadata.get("asset_id") or not metadata.get("training_run_id"):
            return None
        asset = official_asset_for_log(path.parent, self.repository_root, phase='test', metadata=metadata) if self.unified_logs else None
        if self.unified_logs and not asset:
            return None
        parsed = read_paper_result(path.parent, metadata, root)
        relative = _relative(path.parent, root)
        record_id = self._record_id(root, relative, "official")
        pkl = self._prediction_path(path.parent, metadata)
        present = bool(pkl and pkl.is_file() and not pkl.is_symlink())
        metrics = {key: _number(parsed["metrics"].get(key)) for key in METRIC_KEYS}
        samples = metrics.get("samples")
        reported = metadata.get("samples")
        if samples is None and isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            samples = metrics["samples"] = reported
        if samples is not None and float(samples).is_integer():
            samples = int(samples)
        return {"record_id": record_id, "source": "official", "legacy": True,
                "asset_id": asset.get("id") if asset else None,
                "test_id": metadata.get("test_id") or path.parent.name,
                "training_run_id": metadata["training_run_id"],
                "weight_id": metadata.get("weight_id") or "paper:" + str(metadata["asset_id"]),
                "checkpoint": asset.get('path') if asset else metadata.get("checkpoint"),
                "checkpoint_sha256": asset.get('sha256') if asset else None,
                "model": metadata["model"], "action": "evaluate", "task": parsed["task"],
                "split": "test", "split_manifest": parsed["config"].get("split_manifest"),
                "seed": parsed["config"].get("seed"), "config": parsed["config"],
                "dataset_manifest_sha256": None, "split_index_sha256": None,
                "partial": None, "limited": None, "complete": False, "status": "recorded",
                "samples": samples, "expected_samples": None, "joints": metrics.get("joints"),
                "threshold_m": metrics.get("threshold_m"), "metrics": metrics,
                "per_joint": parsed["per_joint"], "paper_tables": asset.get("paper_tables", metadata.get("paper_tables", [])) if asset else metadata.get("paper_tables", []),
                "paper_dependency_tables": asset.get("paper_dependency_tables", metadata.get("paper_dependency_tables", [])) if asset else metadata.get("paper_dependency_tables", []),
                "provenance": metadata.get("provenance", {}),
                "prediction_npz": {"available": False, "format": None, "manifest": None},
                "prediction_pkl": {"present": present, "available": False, "valid": False,
                                   "trusted": self.trusted_pickle,
                                   "pending": bool(present and self.trusted_pickle),
                                   "path": f"{record_id}/{_relative(pkl, path.parent)}" if present else None},
                "_pkl_relative": _relative(pkl, path.parent) if present else None,
                "mtime": (path.parent / "metrics.csv").stat().st_mtime if (path.parent / "metrics.csv").is_file() else path.stat().st_mtime}

    def _relative_checkpoint(self, checkpoint, run_dir):
        value=Path(str(checkpoint))
        candidates=[value] if value.is_absolute() else [run_dir/value,self.repository_root/value]
        for candidate in candidates:
            try:
                resolved=candidate.resolve(strict=True)
                for root in (self.repository_root,*self.roots):
                    if _inside(resolved,root):
                        return resolved.relative_to(root).as_posix()
            except OSError:
                continue
        # Configured external checkpoints are deliberately not exposed as paths.
        return value.name

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _prediction_path(self, directory, info):
        value = info.get("visual_prediction_file") or info.get("prediction_pkl")
        candidates = []
        if isinstance(value, str) and value:
            supplied = Path(value)
            candidates.append(supplied if supplied.is_absolute() else directory / supplied)
        candidates.extend((directory / "test_results.pkl", directory / "predictions" / "test_results.pkl"))
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                continue
            if _inside(resolved,directory.resolve()) and not candidate.is_symlink() and resolved.suffix.lower() in (".pkl", ".pickle"):
                return candidate
        return None

    def _validate_prediction_pickle(self, path):
        with self._lock:
            if path is None or not path.is_file() or path.is_symlink():
                return False
            try:
                if not any(_inside(path.resolve(), root) for root in self.roots) or path.stat().st_size <= 0:
                    return False
                # PKL deserialization is only permitted after explicit operator trust.
                stat = path.stat()
                cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
                if cache_key == self._visual_cache_key and self._visual_cache is not None:
                    if self._visual_cache_validated:
                        return True
                    value = self._visual_cache
                else:
                    self._visual_cache_key, self._visual_cache = None, None
                    self._visual_cache_validated = False
                    self._visual_index_payload = self._visual_index = None
                    with path.open("rb") as stream:
                        value = pickle.load(stream)
                if not isinstance(value, dict):
                    return False
                if not all(key in value for key in ("input", "output", "gt")):
                    return False
                input_shape,input_ok=self._pickle_array_shape(value["input"])
                output_shape,output_ok=self._pickle_array_shape(value["output"])
                target_shape,target_ok=self._pickle_array_shape(value["gt"])
                if not (input_ok and output_ok and target_ok) or len(input_shape)<2 or len(output_shape)<2 or len(target_shape)<2:
                    return False
                if output_shape[0]!=target_shape[0] or (len(input_shape)>=3 and input_shape[0]!=output_shape[0]):
                    return False
                if output_shape[-2:]!=target_shape[-2:] or output_shape[-1:]!=(3,):
                    return False
                stat=path.stat()
                self._visual_cache_key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
                self._visual_cache=value
                self._visual_cache_validated=True
                return True
            except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError, ImportError, AttributeError):
                return False

    @staticmethod
    def _pickle_array_shape(value):
        if isinstance(value, dict):
            return None, False
        if hasattr(value,"detach"):
            value=value.detach().cpu().numpy()
        array=np.asarray(value)
        return tuple(int(dim) for dim in array.shape),array.dtype.kind in "biuf"

    @staticmethod
    def _pickle_array(value):
        return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)

    def _trusted_prediction_payload(self, record):
        with self._lock:
            pathinfo=record.get("prediction_pkl",{})
            if not self.trusted_pickle or not pathinfo.get("present"):
                return None
            manifest=self._record_manifest_path(record["record_id"])
            relative=record.get("_pkl_relative")
            if manifest is None or not relative:
                return None
            path=manifest.parent/relative
            try:
                resolved=path.resolve(strict=True)
                if path.is_symlink() or not _inside(resolved,manifest.parent.resolve()) or not resolved.is_file():
                    return None
                signature=(resolved.stat().st_mtime_ns,resolved.stat().st_size)
                key=(str(resolved),signature[0],signature[1])
                if key==self._visual_cache_key:
                    return self._visual_cache
                with resolved.open("rb") as stream:
                    payload=pickle.load(stream)
                if not isinstance(payload,dict) or not all(k in payload for k in ("input","output","gt")):
                    return None
                self._visual_cache_key,self._visual_cache=key,payload
                self._visual_cache_validated=False
                self._visual_index_payload = self._visual_index = None
                return payload
            except (OSError,EOFError,ValueError,TypeError,pickle.UnpicklingError,ImportError,AttributeError):
                return None

    def _matching_training(self, model, checkpoint_hash, checkpoint, training, root, *, source=None):
        source = source or self._record_source(root, {})
        training = [item for item in training if item["source"] == source]
        for item in training:
            if item["model"] == model and checkpoint_hash and item.get("checkpoint_sha256") == checkpoint_hash:
                return item
        if checkpoint:
            normalized = Path(str(checkpoint)).as_posix().casefold()
            for item in training:
                if item["model"] == model and item.get("checkpoint") and Path(item["checkpoint"]).as_posix().casefold() == normalized:
                    return item
        return None

    def _split_index_hash(self, split_manifest, root, *, config=None, recorded=None):
        if not self.data_root:
            return None
        if config and "split_strategy" in config:
            try:
                from mmwave_densepose.data.partitions import inspect_partition
                current = inspect_partition(self.data_root, config)
                metadata = current["metadata"]
                if current["pending"]:
                    return None
                expected = metadata.get("index_sha256")
                recorded = recorded or {}
                if recorded.get("split_index_sha256") not in (None, expected):
                    return None
                if recorded.get("partition_cache_id") not in (None, current["cache_id"]):
                    return None
                return expected
            except (OSError, ValueError, KeyError, TypeError):
                return None
        if not split_manifest:
            return None
        source=self.data_root
        try:
            source_manifest=source/"manifest.json"
            if not source_manifest.is_file() or source_manifest.is_symlink():
                return None
            manifest=_read_json(source_manifest) or {}
            if manifest.get("dataset_kind")=="full_multimodal":
                cache=source/"cache"/"pose-v1"
                cache_manifest_path=cache/"manifest.json"
                if not cache_manifest_path.is_file() or cache_manifest_path.is_symlink():
                    return None
                cache_manifest=_read_json(cache_manifest_path) or {}
                if cache_manifest.get("source_manifest_sha256")!=self._sha256(source_manifest):
                    return None
                source=cache
            split_path=source/"splits"/(Path(str(split_manifest)).stem+".json")
            index_path=split_path.with_suffix(".npz")
            resolved=split_path.resolve(strict=True);resolved_index=index_path.resolve(strict=True)
            if (not _inside(resolved,source.resolve()) or split_path.is_symlink() or
                    not _inside(resolved_index,source.resolve()) or index_path.is_symlink()):
                return None
            metadata=_read_json(resolved) or {};expected=metadata.get("index_sha256")
            if not expected or expected!=self._sha256(resolved_index):
                return None
            return expected
        except OSError:
            return None

    def _per_joint(self, directory, root):
        files = (directory / "per_joint_metrics.csv", directory / "per_cloud_metrics.csv")
        for path in files:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
                if not _inside(resolved, root):
                    continue
                with path.open("r", newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream)
                    rows = []
                    for row in reader:
                        raw_index = row.get("joint_index")
                        if raw_index is None:
                            raw_index = row.get("cloud_index")
                        try:
                            index = int(raw_index)
                        except (TypeError, ValueError):
                            continue
                        metrics={k:_csv_number(row.get(k)) for k in
                                 ("mle_m","mle_cm","pa_mpjpe_m","pa_mpjpe_cm","pck","pck_percent","auc","auc_percent","cd_m","cd_cm")}
                        for metres,centimetres in (("mle_m","mle_cm"),("pa_mpjpe_m","pa_mpjpe_cm"),("cd_m","cd_cm")):
                            if metrics[centimetres] is None and metrics[metres] is not None:
                                metrics[centimetres]=metrics[metres]*100
                        for fraction,percentage in (("pck","pck_percent"),("auc","auc_percent")):
                            if metrics[percentage] is None and metrics[fraction] is not None:
                                metrics[percentage]=metrics[fraction]*100
                        rows.append({"index":index,**metrics})
                    if rows:
                        rows.sort(key=lambda r: r["index"])
                        return rows
            except (OSError, csv.Error, UnicodeError):
                continue
        return None

    def _weight_id(self, row):
        return row.get("weight_id") or row.get("checkpoint_sha256") or row.get("checkpoint")

    def _source_records(self, source, *, refresh=True):
        if source not in ("user", "official"):
            raise ValueError("Result source must be user or official")
        # Playback already selected a discovered run. Do not rescan all logs
        # (and hash checkpoints) on a frame's critical path; list/refresh calls
        # retain normal discovery, while frame files are checked independently.
        snapshot = self._records
        if refresh or snapshot is None:
            snapshot = self._load_records()
        return {kind: [row for row in rows if row["source"] == source]
                for kind, rows in snapshot.items()}

    def _protected_test_paths(self):
        """Protect paper-designated paths even if their per-test sidecar is lost."""
        from .log_index import LogIndex
        manifest = self.repository_root / "paper/manifest.json"
        if not manifest.exists():
            return set()
        if not LogIndex._no_links(manifest, self.repository_root):
            raise PermissionError("Official evaluation protection metadata is unavailable.")
        data = _read_json(manifest)
        if not data or not isinstance(data.get("assets"), list):
            raise PermissionError("Official evaluation protection metadata is unavailable.")
        protected = set()
        for asset in data["assets"]:
            if not isinstance(asset, dict):
                continue
            for evaluation in asset.get("evaluations", []):
                value = evaluation.get("path") if isinstance(evaluation, dict) else evaluation
                if not isinstance(value, str):
                    continue
                relative = Path(value)
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                path = (self.repository_root / relative).resolve()
                if path != self.outputs_root and _inside(path, self.outputs_root):
                    protected.add(path)
        return protected

    def _deletion_directory(self, record, protected=None):
        from .log_index import LogIndex
        if record.get("source") != "user" or record.get("action") not in {"evaluate", "test"}:
            raise PermissionError("Only personal evaluation logs can be deleted.")
        expected = self.repository_root / "mmwave_densepose/logs/test"
        if self.outputs_root != expected:
            raise PermissionError("Only individual records in the evaluation-log directory can be deleted.")
        manifest = self._record_manifest_path(record["record_id"])
        if manifest is None:
            raise PermissionError("The evaluation directory is unavailable.")
        directory = manifest.parent
        relative = directory.relative_to(expected)
        if (len(relative.parts) != 3 or relative.parts[0] != record.get("model")
                or relative.parts[-1] != record.get("test_id")):
            raise PermissionError("Select an individual test record, not its model or training directory.")
        if not LogIndex._no_links(directory, self.repository_root) or not directory.is_dir():
            raise PermissionError("The evaluation directory contains a linked path or is unavailable.")
        protected = self._protected_test_paths() if protected is None else protected
        if ((directory / "paper-record.json").exists() or (directory / "paper-training.json").exists()
                or any(_inside(directory, path) or _inside(path, directory) for path in protected)):
            raise PermissionError("Official evaluation logs cannot be deleted.")
        return directory

    def delete_records(self, record_ids, *, active_paths=()):
        """Preflight the complete batch, then move only its test leaves to recovery storage."""
        from .log_index import LogIndex
        if (not isinstance(record_ids, (list, tuple)) or not 1 <= len(record_ids) <= 500
                or any(not isinstance(value, str) or not 1 <= len(value) <= 2048
                       or ":" not in value or "\\" in value or "\x00" in value
                       or ".." in Path(value.partition(":")[2]).parts for value in record_ids)
                or len(set(record_ids)) != len(record_ids)):
            raise ValueError("Select one to 500 distinct indexed evaluation records.")
        with self._lock:
            snapshot = self._load_records(refresh=True)
            indexed = {record["record_id"]: record for record in snapshot["tests"]}
            if any(record_id not in indexed for record_id in record_ids):
                raise KeyError("Evaluation record not found; refresh the result list.")
            records = [indexed[record_id] for record_id in record_ids]
            protected = self._protected_test_paths()
            directories = [self._deletion_directory(record, protected) for record in records]
            def walk_error(error):
                raise error
            for directory in directories:
                for base, folders, files in os.walk(directory, followlinks=False, onerror=walk_error):
                    for name in (*folders, *files):
                        path = Path(base) / name
                        if not LogIndex._no_links(path, self.repository_root):
                            raise PermissionError("Evaluation logs containing linked paths cannot be deleted.")
                        if (name in {"paper-record.json", "paper-training.json", "training_summary.json"}
                                or name == "run.json" and path.parent != directory):
                            raise PermissionError("This directory contains another evaluation or training record.")
                        if ((path.is_dir() and name in {"train", "training", "checkpoints", "evaluations", "tests"})
                                or path.is_file() and path.suffix.casefold() in {".ckpt", ".pt", ".pth", ".safetensors"}):
                            raise PermissionError("This directory contains training logs or model weights.")
            for position, directory in enumerate(directories):
                if any(_inside(directory, other) or _inside(other, directory) for other in directories[:position]):
                    raise ValueError("Selected evaluation directories overlap.")
            monitor = LogIndex(workspace_root=self.repository_root, actions={"train"})
            active = monitor._active_training_paths(directories)
            for value in active_paths:
                path = Path(value)
                active.append((path if path.is_absolute() else self.repository_root / path).resolve())
            if any(_inside(directory, path) or _inside(path, directory) for directory in directories for path in active):
                raise RuntimeError("Stop the selected evaluation tasks before deleting their logs.")
            trash = self.repository_root / ".local/trash/results"
            ancestor = trash
            while not ancestor.exists():
                ancestor = ancestor.parent
            if not LogIndex._no_links(ancestor, self.repository_root):
                raise PermissionError("Evaluation recovery storage contains a linked path.")
            trash.mkdir(parents=True, exist_ok=True)
            if not LogIndex._no_links(trash, self.repository_root):
                raise PermissionError("Evaluation recovery storage contains a linked path.")
            batch = trash / uuid.uuid4().hex
            batch.mkdir()
            entries = [{"record_id": record["record_id"],
                        "source": directory.relative_to(self.repository_root).as_posix(),
                        "destination": f"records/{number:03d}"}
                       for number, (record, directory) in enumerate(zip(records, directories), 1)]
            recovery = {"schema_version": 1, "status": "planned", "created_at": time.time(), "entries": entries}
            manifest_path = batch / "manifest.json"
            manifest_path.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
            (batch / "records").mkdir()
            moved = []
            try:
                for record, directory, entry in zip(records, directories, entries):
                    if self._deletion_directory(record, self._protected_test_paths()) != directory:
                        raise PermissionError("An evaluation path changed; refresh the result list.")
                    destination = batch / entry["destination"]
                    directory.rename(destination)
                    moved.append((directory, destination))
                recovery["status"] = "deleted"
                manifest_path.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
            except OSError:
                failed = []
                for directory, destination in reversed(moved):
                    try:
                        destination.rename(directory)
                    except OSError:
                        failed.append(directory.relative_to(self.repository_root).as_posix())
                recovery.update(status="recovery_required" if failed else "rolled_back", recovery_required=failed)
                manifest_path.write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
                if failed:
                    raise RuntimeError(f"Recover evaluation records using {manifest_path.relative_to(self.repository_root).as_posix()}.") from None
                raise
            finally:
                self._records = None
                self._records_at = 0
                self._visual_cache_key = self._visual_cache = None
                self._visual_cache_validated = False
                self._visual_index_payload = self._visual_index = None
                self._visual_requested_records.difference_update(record_ids)
            return {"deleted": list(record_ids), "count": len(record_ids), "trash_id": batch.name,
                    "recovery_manifest": manifest_path.relative_to(self.repository_root).as_posix()}

    def list_models(self, *, source="user"):
        snapshot = self._source_records(source)
        models = sorted({r["model"] for r in snapshot["tests"]} | {r["model"] for r in snapshot["training"]}, key=str.casefold)
        return {"source": source, "models": [{"id": model, "name": model, "source": source,
                            "weight_count": len({self._weight_id(r) for r in snapshot["tests"] + snapshot["training"] if r["model"] == model}),
                            "test_count": sum(r["model"] == model for r in snapshot["tests"])} for model in models]}

    def list_weights(self, model, *, source="user"):
        snapshot = self._source_records(source)
        rows = [r for r in snapshot["tests"] + snapshot["training"] if r["model"] == model]
        weights = {}
        for row in rows:
            weight = self._weight_id(row)
            if not weight:
                continue
            item = weights.setdefault(weight, {"id": weight, "model": model, "source": source,
                "training_run_id": row.get("training_run_id"), "checkpoint": row.get("checkpoint"),
                "checkpoint_sha256": row.get("checkpoint_sha256"), "training_config": self._public_config(row.get("config", {})),
                "test_count": 0, "tests": []})
            if row.get('action') == 'train':
                item['training_config'] = self._public_config(row.get('config', {}))
            item['paper_tables'] = row.get('paper_tables', item.get('paper_tables', []))
            item['paper_dependency_tables'] = row.get('paper_dependency_tables', item.get('paper_dependency_tables', []))
            if row.get("action") in ("evaluate", "test"):
                item["test_count"] += 1
                item["tests"].append({"record_id": row["record_id"], "source": source, "test_id": row["test_id"],
                    "status": row["status"], "mtime": row["mtime"],
                    "paper_tables": row.get('paper_tables', []), "paper_dependency_tables": row.get('paper_dependency_tables', []),
                    "visualization_available": row["prediction_pkl"]["available"]})
        result = sorted(weights.values(), key=lambda r: (r.get("training_run_id") or "", r["id"]))
        return {"source": source, "model": model, "weights": result}

    def list_tests(self, model, training_run_id=None, weight_id=None, *, source="user"):
        snapshot = self._source_records(source)
        rows = [r for r in snapshot["tests"] if r["model"] == model]
        if training_run_id is not None:
            rows = [r for r in rows if r["training_run_id"] == training_run_id]
        if weight_id is not None:
            rows = [r for r in rows if self._weight_id(r) == weight_id]
        return {"source": source, "model": model, "training_run_id": training_run_id,
                "weight_id": weight_id, "tests": [self._public_record(r) for r in rows]}

    def _public_config(self, config):
        safe={}
        for key,value in config.items():
            if key in ("data_root","checkpoint","completion_checkpoint") and isinstance(value,str):
                candidate=Path(value)
                if candidate.is_absolute():
                    try:value=candidate.resolve().relative_to(self.repository_root).as_posix()
                    except (ValueError,OSError):value=candidate.name
            safe[key]=value
        return safe

    def _public_record(self, record):
        payload={key:record.get(key) for key in (
            "record_id", "source", "asset_id", "legacy", "task", "test_id", "training_run_id", "weight_id", "model", "action", "split",
            "split_manifest", "split_strategy", "partition_cache_id", "seed", "dataset_manifest_sha256", "split_index_sha256", "partial",
            "limited", "complete", "status", "visual_requested", "samples", "expected_samples", "joints",
            "threshold_m", "metrics", "per_joint", "prediction_npz", "prediction_pkl", "mtime", "paper_tables", "paper_dependency_tables", "provenance")}
        payload["config"]=self._public_config(record.get("config",{}))
        try:
            self._deletion_directory(record)
            payload["can_delete"] = True
        except (OSError, ValueError, PermissionError):
            payload["can_delete"] = False
        return payload

    def get_record(self, record_id, *, source="user", _refresh=True):
        for record in self._source_records(source,refresh=_refresh)["tests"]:
            if record["record_id"] == record_id:
                pkl = record["prediction_pkl"]
                if pkl["present"] and self.trusted_pickle and (not record.get("legacy") or record_id in self._visual_requested_records):
                    manifest = self._record_manifest_path(record_id)
                    path = manifest.parent / record["_pkl_relative"] if manifest and record.get("_pkl_relative") else None
                    try:
                        resolved=path.resolve(strict=True) if path else None
                        path=path if resolved and _inside(resolved,manifest.parent.resolve()) and not path.is_symlink() else None
                    except OSError:
                        path=None
                    signature = ((path.stat().st_mtime_ns, path.stat().st_size) if path and path.is_file() else None)
                    if '_pkl_valid' not in record or record.get("_pkl_signature") != signature:
                        record["_pkl_valid"] = self._validate_prediction_pickle(path)
                        record["_pkl_signature"] = signature
                        if record["_pkl_valid"]:
                            manifest = self._record_manifest_path(record_id)
                            relative = record.get("_pkl_relative")
                            if manifest and relative:
                                self._trusted_prediction_payload({**record,"_pkl_relative":relative})
                    pkl["valid"] = bool(record["_pkl_valid"])
                    pkl["available"] = bool(record["_pkl_valid"])
                    pkl["pending"] = False
                return self._public_record(record)
        return None

    @staticmethod
    def _comparison_issues(records,threshold):
        issues = []
        if len(records) < 2:
            return ["Select at least two test records."]
        if any(not r.get("dataset_manifest_sha256") or not r.get("split_index_sha256") for r in records):
            issues.append("unverified dataset or split identity")
        keys = ("dataset_manifest_sha256", "split_index_sha256", "split_strategy", "split_manifest", "split", "joints", "threshold_m")
        for key in keys:
            values = {json.dumps(r.get(key), sort_keys=True) for r in records}
            if len(values) > 1:
                issues.append(key.replace("_", " ")+" mismatch")
        for record in records:
            if not record["complete"]:
                issues.append(f"{record['test_id']} has unverified completeness" if record.get("legacy") else f"{record['test_id']} is incomplete")
        for key in CONFIG_KEYS:
            values={json.dumps(r.get("config",{}).get(key),sort_keys=True) for r in records}
            if len(values)>1:
                issues.append(key.replace("_"," ")+" mismatch")
        if any(_number(r.get("threshold_m")) is None or abs(float(r["threshold_m"])-float(threshold))>1e-8 for r in records):
            issues.append("requested metric threshold mismatch")
        return issues

    def compare(self, record_ids, threshold=0.15, *, source="user"):
        threshold_value=_number(threshold)
        if threshold_value is None or threshold_value<=0:
            raise ValueError("Comparison threshold must be positive")
        lookup = {r["record_id"]: r for r in self._source_records(source)["tests"]}
        if any(record_id not in lookup for record_id in record_ids):
            raise ValueError("Test record not found in selected source")
        records = [self._public_record(lookup[record_id]) for record_id in dict.fromkeys(record_ids)]
        issues = self._comparison_issues(records,threshold_value)
        samples = [self.sample_map(r["record_id"], source=source) for r in records]
        shared = set(samples[0]) if samples else set()
        for sample_ids in samples[1:]:
            shared.intersection_update(sample_ids)
        canonical = {sample_id for sample_id in shared if parse_sample_id(sample_id)}
        return {"source": source, "records": records, "compatible": not issues, "issues": list(dict.fromkeys(issues)),
                "synchronization_allowed": not issues and bool(canonical), "shared_samples": len(canonical),
                "threshold_m": float(threshold), "sample_ids": sorted(canonical)[:1000]}

    def _pickle_sample_index(self, payload):
        # Keep one index, matching the single active PKL cache. In particular,
        # do not recreate all IDs or convert all frame arrays during playback.
        with self._lock:
            if self._visual_index_payload is payload:
                return self._visual_index
            ids=payload.get("sample_id")
            if ids is None:
                meta=payload.get("meta")
                ids=meta.get("sample_ids") if isinstance(meta,dict) else None
            if hasattr(ids,"tolist"):
                ids=ids.tolist()
            count=len(payload["output"])
            if ids is None:
                output=[f"pkl-index:{i}" for i in range(count)]
            else:
                output=[str(value) for value in ids]
                if len(output)!=count or len(set(output))!=count:
                    output=[]
            lookup={value:i for i,value in enumerate(output)}
            parsed={value:parse_sample_id(value) for value in output}
            def order(value):
                info=parsed[value]
                if info:
                    return (info['sequence_id'],info['start'],info['end'])
                if value.startswith("pkl-index:") and value[10:].isdigit():
                    return ('',int(value[10:]),'')
                return ('',0,value)
            ordered=sorted(output,key=order)
            sequences={}
            for value in ordered:
                info=parsed[value]
                if info:
                    sequences.setdefault(info['sequence_id'],[]).append(value)
            index={"ids":output,"lookup":lookup,"ordered":ordered,
                   "parsed":parsed,"sequences":sequences,
                   "sequence_counts":{key:len(values) for key,values in sequences.items()}}
            self._visual_index_payload,self._visual_index=payload,index
            return index

    def _pickle_sample_ids(self, record, payload):
        # Legacy indexed frames remain viewable but do not become canonical IDs.
        return self._pickle_sample_index(payload)["ids"]

    def sample_map(self, record_id, *, source="user"):
        record = next((r for r in self._source_records(source,refresh=False)["tests"] if r["record_id"] == record_id), None)
        if not record:
            return set()
        if record.get("legacy") and record_id not in self._visual_requested_records:
            return set()
        if record["prediction_pkl"].get("present") and self.trusted_pickle:
            selected=self.get_record(record_id, source=source,_refresh=False)
            if selected and selected["prediction_pkl"].get("available"):
                payload=self._trusted_prediction_payload(record)
                if payload is not None:
                    return set(self._pickle_sample_ids(record,payload))
        manifest_path = self._record_manifest_path(record_id)
        if manifest_path is None:
            return set()
        pred_dir = manifest_path.parent / "predictions"
        manifest = _read_json(pred_dir / "manifest.json") or {}
        output = set()
        for batch in manifest.get("batches", []):
            filename = batch.get("file") if isinstance(batch, dict) else None
            if not filename or Path(filename).name != filename or Path(filename).suffix != ".npz":
                continue
            path = (pred_dir / filename)
            try:
                resolved = path.resolve(strict=True)
                if not _inside(resolved, manifest_path.parent) or resolved.is_symlink():
                    continue
                with np.load(resolved, allow_pickle=False) as arrays:
                    if "sample_id" not in arrays.files:
                        continue
                    output.update(str(x) for x in arrays["sample_id"].tolist() if parse_sample_id(x))
            except (OSError, ValueError, KeyError):
                continue
        return output

    def _record_manifest_path(self, record_id):
        for root in self.roots:
            try:
                prefix, separator, remainder = str(record_id).partition(":")
                expected_roots = {"logs-user", "logs-official"} if self.unified_logs else {"outputs" if root == self.outputs_root else "reproduce-results"}
                if not separator or prefix not in expected_roots:
                    continue
                relative = Path(remainder)
                if relative.is_absolute() or ".." in relative.parts:
                    return None
                directory = root / relative
                names = ('paper-record.json', 'run.json') if self.unified_logs and prefix == 'logs-official' else ('run.json', 'paper-record.json')
                for name in names:
                    path = directory / name
                    if not path.is_file():
                        continue
                    resolved = path.resolve(strict=True)
                    if _inside(resolved, root) and not path.is_symlink():
                        if self.unified_logs:
                            info = _read_json(path) or {}
                            origin = self._record_source(root, info, directory)
                            if prefix != "logs-" + origin:
                                continue
                        return path
            except (OSError, ValueError):
                continue
        return None

    def samples(self, record_id, limit=100, offset=0, sequence_id=None, *, source="user"):
        if any(row["record_id"] == record_id for row in self._source_records(source,refresh=False)["tests"]):
            self._visual_requested_records.add(record_id)
        record=next((row for row in self._source_records(source,refresh=False)["tests"] if row["record_id"]==record_id),None)
        cached=None
        if record and record["prediction_pkl"].get("present") and self.trusted_pickle:
            selected=self.get_record(record_id,source=source,_refresh=False)
            payload=self._trusted_prediction_payload(record) if selected and selected["prediction_pkl"].get("available") else None
            if payload is not None:
                cached=self._pickle_sample_index(payload)
        if cached is not None:
            ids=cached["ordered"] if sequence_id is None else cached["sequences"].get(sequence_id,[])
            offset,limit=max(0,int(offset)),min(500,max(1,int(limit)))
            items=[cached["parsed"][value] or {"sample_id":value,"sequence_id":None,
                   "start":None,"end":None,"target_frame":None} for value in ids[offset:offset+limit]]
            return {"source":source,"record_id":record_id,"samples":items,
                    "total":len(ids),"total_all":len(cached["ordered"]),"offset":offset,"limit":limit,
                    "sequence_ids":sorted(cached["sequences"]),"sequence_counts":cached["sequence_counts"],
                    "sequence_id":sequence_id}
        def order(value):
            parsed=parse_sample_id(value)
            if parsed:
                return (parsed['sequence_id'], parsed['start'], parsed['end'])
            if str(value).startswith("pkl-index:") and str(value)[10:].isdigit():
                return ('', int(str(value)[10:]), '')
            return ('', 0, str(value))
        ids = sorted(self.sample_map(record_id, source=source),key=order)
        sequence_counts={}
        for value in ids:
            parsed=parse_sample_id(value)
            if parsed:
                key=parsed['sequence_id'];sequence_counts[key]=sequence_counts.get(key,0)+1
        total_all=len(ids)
        if sequence_id is not None:
            ids=[value for value in ids if (parse_sample_id(value) or {}).get('sequence_id')==sequence_id]
        offset, limit = max(0, int(offset)), min(500, max(1, int(limit)))
        chosen = ids[offset:offset + limit]
        items=[]
        for sample_id in chosen:
            parsed=parse_sample_id(sample_id)
            items.append(parsed if parsed is not None else
                         {"sample_id":sample_id,"sequence_id":None,"start":None,"end":None,"target_frame":None})
        return {"source": source, "record_id": record_id, "samples": items,
                "total": len(ids), "total_all":total_all, "offset": offset, "limit": limit,
                "sequence_ids":sorted(sequence_counts),"sequence_counts":sequence_counts,
                "sequence_id":sequence_id}

    def sample(self, record_id, sample_id, *, source="user"):
        sample = parse_sample_id(sample_id)
        record = next((r for r in self._source_records(source,refresh=False)["tests"] if r["record_id"] == record_id), None)
        if not record:
            return None
        self._visual_requested_records.add(record_id)
        if record["prediction_pkl"].get("present") and self.trusted_pickle:
            selected=self.get_record(record_id, source=source,_refresh=False)
            payload=self._trusted_prediction_payload(record) if selected and selected["prediction_pkl"].get("available") else None
            if payload is not None:
                index=self._pickle_sample_index(payload)["lookup"].get(str(sample_id),-1)
                if index>=0:
                    inputs=self._pickle_array(payload["input"][index])
                    prediction=self._pickle_array(payload["output"][index])
                    target=self._pickle_array(payload["gt"][index])
                    radar=inputs
                    if radar.ndim==4:radar=radar[-1]
                    elif radar.ndim==3:radar=radar[-1]
                    info=parse_sample_id(sample_id) or {"sample_id":str(sample_id),"sequence_id":None,"start":None,"end":None,"target_frame":None}
                    return {**info,"source":source,"radar":_safe_json(radar),"prediction":_safe_json(prediction),
                            "target":_safe_json(target),"joints":record["joints"],
                            "coordinate_frame":"stored_world_z_up","xyz_units":"m",
                            "window_input":"last frame" if inputs.ndim>=3 else "target frame"}
        if not sample or sample_id not in self.sample_map(record_id, source=source):
            return None
        if record["prediction_pkl"].get("available"):
            return None
        # Numeric predictions stay sharded: load only the batch containing the requested ID.
        manifest_path = self._record_manifest_path(record_id)
        pred_dir = manifest_path.parent / "predictions"
        manifest = _read_json(pred_dir / "manifest.json") or {}
        for batch in manifest.get("batches", []):
            filename = batch.get("file") if isinstance(batch, dict) else None
            if not filename or Path(filename).name != filename:
                continue
            path = pred_dir / filename
            try:
                resolved = path.resolve(strict=True)
                if not _inside(resolved, manifest_path.parent) or resolved.is_symlink():
                    continue
                with np.load(resolved, allow_pickle=False) as arrays:
                    ids = [str(x) for x in arrays["sample_id"].tolist()]
                    try:
                        index = ids.index(sample_id)
                    except ValueError:
                        continue
                    prediction, target = arrays["prediction"], arrays["target"]
                    return {**sample, "source":source, "prediction": _safe_json(prediction[index]),
                            "target": _safe_json(target[index]), "joints": record["joints"],
                            "coordinate_frame": "stored_world_z_up", "xyz_units": "m",
                            "source_points": None}
            except (OSError, ValueError, KeyError, IndexError):
                continue
        return None
