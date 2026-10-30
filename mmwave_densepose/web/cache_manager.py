"""Read cache descriptors and recoverably remove generated partition files."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
import uuid

from mmwave_densepose.data.partitions import _lock


_CACHE_ID = re.compile(r"[0-9a-f]{64}\Z")
_ACTIVE = {"Queued", "Starting", "Running", "Stopping"}
_PARAMETERS = (
    "model", "task", "data_family", "split_strategy", "seed", "partition",
    "split_chunk_size", "max_timestamp_gap_ms", "min_valid_points", "num_points",
    "input_dim", "num_joints", "seq_len", "num_out_points", "pcc_label_radius",
)
_ALIASES = {"sample_dim": "input_dim", "feature_dim": "input_dim",
            "num_keypoints": "num_joints", "radar_point_size": "num_points", "t_max": "seq_len"}
_UNAFFECTED = ["batch_size", "epochs", "learning_rate", "weight_decay", "device", "gpu_ids", "num_devices", "num_workers", "checkpoint"]
_LIMIT = 16 * 1024 * 1024


def _atomic_bytes(path, raw):
    descriptor, temporary = tempfile.mkstemp(prefix=".cache-manager-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class CacheManager:
    """Manage derived partitions only; source and shared sequence arrays are protected."""

    def __init__(self, data_root, repository_root, active_jobs=None, *, lock_timeout=.2):
        self.data_root = Path(data_root).resolve()
        self.repository_root = Path(repository_root).resolve()
        self.active_jobs = active_jobs
        self.lock_timeout = float(lock_timeout)
        self._mutex = threading.RLock()

    def _path(self, relative, *, missing=False):
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts or not value.parts:
            raise ValueError("Invalid dataset cache path")
        candidate = self.data_root / value
        current = self.data_root
        for part in value.parts:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                if missing:
                    continue
                raise ValueError("Cache artifact is missing") from None
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise PermissionError("Cache paths must not contain links")
        if not candidate.resolve().is_relative_to(self.data_root):
            raise PermissionError("Cache path is outside the configured dataset")
        return candidate

    def _json(self, relative, *, optional=False):
        try:
            path = self._path(relative)
            if not path.is_file() or path.stat().st_size > _LIMIT:
                raise ValueError("Cache metadata is unavailable or too large")
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Cache metadata must contain an object")
            return value, raw
        except (OSError, ValueError):
            if optional:
                return {}, None
            raise ValueError("Cache metadata could not be read") from None

    def _registry(self):
        value, raw = self._json("cache/index.json", optional=True)
        if raw is None:
            # Reading an absent index does not create it or materialize caches.
            if self._path("cache/index.json", missing=True).exists():
                raise ValueError("The cache index could not be read")
            return {"schema_version": 2, "entries": {}}, None
        if not isinstance(value.get("entries"), dict):
            raise ValueError("The cache index has an invalid entry list")
        return value, raw

    @staticmethod
    def _parameters(entry, metadata):
        origin = entry.get("original_provenance", {})
        origin = origin if isinstance(origin, dict) else {}
        values = {}
        for source in (origin, origin.get("config", {}), metadata, metadata.get("config", {}), entry.get("parameters", {})):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                canonical = _ALIASES.get(key, key)
                if canonical in _PARAMETERS:
                    values[canonical] = value
        return values

    def _sizes(self, relatives):
        size, count = 0, 0
        seen = set()
        for relative in relatives:
            if not isinstance(relative, str) or relative in seen:
                continue
            seen.add(relative)
            try:
                path = self._path(relative)
                if path.is_file():
                    size += path.stat().st_size
                    count += 1
            except (OSError, ValueError, PermissionError):
                pass
        return size, count

    def _delete_reason(self, cache_id, entry):
        if not _CACHE_ID.fullmatch(cache_id) or entry.get("kind") != "partition":
            return "Only generated partition caches can be deleted."
        directory = f"cache/partitions/{cache_id}"
        if (entry.get("metadata_path") != directory + "/splits.json" or
                entry.get("index_path") != directory + "/splits.npz"):
            return "The cache index does not identify a generated partition directory."
        try:
            path = self._path(directory)
            if not path.is_dir():
                return "The partition directory is missing."
            for child in path.iterdir():
                self._path(child.relative_to(self.data_root))
                if not child.is_file() or child.name not in {"splits.json", "splits.npz", ".build.lock"}:
                    return "The partition directory contains unrecognized files."
        except (OSError, ValueError, PermissionError):
            return "The partition directory is unavailable or contains linked paths."
        return None

    def _row(self, cache_id, entry, source_hash):
        if not isinstance(cache_id, str) or not isinstance(entry, dict):
            return None
        relative = entry.get("metadata_path")
        metadata, raw = ({}, None)
        unsafe = False
        if isinstance(relative, str):
            try:
                metadata, raw = self._json(relative, optional=True)
            except PermissionError:
                unsafe = True
        parameters = self._parameters(entry, metadata)
        origin = entry.get("original_provenance", {})
        origin = origin if isinstance(origin, dict) else {}
        samples = metadata.get("split_num_samples", entry.get("split_num_samples", origin.get("split_num_samples")))
        files = [entry.get("metadata_path"), entry.get("index_path")]
        size, count = self._sizes(files)
        generated = entry.get("kind") == "partition"
        status = entry.get("status", "prepared" if not generated else "ready")
        if unsafe:
            status = "unavailable"
        elif raw is None or (entry.get("index_path") and count < 2):
            status = "missing"
        elif entry.get("metadata_sha256") and hashlib.sha256(raw).hexdigest() != entry["metadata_sha256"]:
            status = "changed"
        expected_source = entry.get("source_manifest_sha256") or metadata.get("source_manifest_sha256") or origin.get("source_manifest_sha256")
        if expected_source and source_hash and expected_source != source_hash:
            status = "source_changed"
        reason = self._delete_reason(cache_id, entry)
        labels = {"partition": "Partition cache", "orphan_partition": "Unregistered partition",
                  "legacy_configuration": "Prepared configuration", "legacy_partition": "Prepared partition",
                  "legacy_manifest": "Prepared sequences"}
        raw_kind = entry.get("kind", "prepared")
        public_kind = raw_kind if raw_kind in {"partition", "orphan_partition"} else "prepared"
        name = " · ".join(str(parameters[key]) for key in ("model", "split_strategy") if parameters.get(key))
        if parameters.get("seed") is not None:
            name += (" · " if name else "") + f"seed {parameters['seed']}"
        return {
            "id": cache_id, "cache_id": cache_id, "kind": public_kind,
            "label": name or labels.get(raw_kind, "Prepared cache"), "cache_type": labels.get(raw_kind, "Prepared cache"),
            "status": status, "path": relative, "index_path": entry.get("index_path"),
            "size_bytes": size, "file_count": count, "parameters": parameters,
            "split_num_samples": samples if isinstance(samples, dict) else {},
            "can_delete": reason is None, "delete_reason": reason,
            "algorithm_version": metadata.get("algorithm_version", entry.get("algorithm_version")),
            "source_manifest_sha256": expected_source,
            "source_inventory_sha256": metadata.get("source_inventory_sha256", entry.get("source_inventory_sha256")),
            "index_sha256": entry.get("index_sha256", metadata.get("index_sha256")),
            "numeric_root": entry.get("numeric_root"),
            "sequence_count": len(metadata["sequence_ids"]) if isinstance(metadata.get("sequence_ids"), list) else None,
            "unaffected_by": list(_UNAFFECTED), "integrity_check": "metadata_and_paths",
        }

    def _shared(self, registry, source_hash):
        shared = registry.get("shared_sequences", {})
        shared = shared if isinstance(shared, dict) else {}
        relative = shared.get("root", "cache/sequences-v2")
        # Shared numeric data is a separate protected entry, never an arbitrary root.
        if relative != "cache/sequences-v2":
            return None
        manifest, raw = self._json(relative + "/manifest.json", optional=True)
        if not raw and not shared:
            return None
        sequences = manifest.get("sequences", {})
        sequences = sequences if isinstance(sequences, dict) else {}
        files = [relative + "/manifest.json"]
        files.extend(relative + "/" + value["file"] for value in sequences.values()
                     if isinstance(value, dict) and isinstance(value.get("file"), str))
        size, count = self._sizes(files)
        source = manifest.get("source_manifest_sha256", shared.get("source_manifest_sha256"))
        status = "ready" if raw is not None and count == len(files) else "missing"
        if source and source_hash and source != source_hash:
            status = "source_changed"
        return {"id": "shared_sequences", "cache_id": "shared_sequences", "kind": "shared_sequences",
                "label": "Shared numeric sequences", "cache_type": "Shared sequence data", "status": status,
                "path": relative, "size_bytes": size, "file_count": count, "sequence_count": len(sequences),
                "parameters": {"contents": ["radar points", "frame offsets", "19-joint labels", "timestamps"],
                               "point_sampling": "unsampled", "timestamps_available": manifest.get("timestamps_available")},
                "split_num_samples": {}, "source_manifest_sha256": source,
                "can_delete": False, "delete_reason": "Shared numeric sequences are used by multiple partition caches.",
                "unaffected_by": [*_UNAFFECTED, "model", "split_strategy", "seed", "seq_len", "num_points", "input_dim"],
                "integrity_check": "metadata_and_paths"}

    def inventory(self):
        with self._mutex:
            registry, _raw = self._registry()
            _source, source_raw = self._json("manifest.json", optional=True)
            source_hash = hashlib.sha256(source_raw).hexdigest() if source_raw else None
            rows = [self._row(cache_id, entry, source_hash) for cache_id, entry in registry["entries"].items()]
            rows = [row for row in rows if row is not None]
            shared = self._shared(registry, source_hash)
            if shared:
                rows.append(shared)
            rows.sort(key=lambda row: (row["kind"] != "partition", row["label"], row["id"]))
            return {"entries": rows, "total": len(rows), "index_path": "cache/index.json",
                    "data_root": self.data_root.relative_to(self.repository_root).as_posix()
                    if self.data_root.is_relative_to(self.repository_root) else self.data_root.name}

    def detail(self, cache_id):
        if not isinstance(cache_id, str) or len(cache_id) > 160:
            raise ValueError("Invalid cache identifier")
        for entry in self.inventory()["entries"]:
            if entry["id"] == cache_id:
                return entry
        raise KeyError("Cache entry not found")

    def _check_active(self):
        if not callable(self.active_jobs):
            raise RuntimeError("Training activity cannot be checked; cache deletion is unavailable.")
        try:
            jobs = self.active_jobs()
            if not isinstance(jobs, (list, tuple)):
                raise TypeError()
        except Exception as error:
            raise RuntimeError("Training activity could not be checked; try again later.") from error
        for job in jobs:
            if not isinstance(job, dict) or job.get("status") not in _ACTIVE:
                continue
            config = job.get("config") if isinstance(job.get("config"), dict) else {}
            value = config.get("data_root")
            if not isinstance(value, str) or not value:
                raise RuntimeError("Stop active training and testing before deleting dataset caches.")
            if value.startswith("configured-data-root/"):
                raise RuntimeError("Stop active training and testing before deleting dataset caches.")
            path = Path(value)
            path = (path if path.is_absolute() else self.repository_root / path).resolve()
            if path == self.data_root:
                raise RuntimeError("Stop active training and testing before deleting dataset caches.")

    def delete(self, cache_ids):
        """Move selected partition payloads to dataset-local recovery storage."""
        if (not isinstance(cache_ids, list) or not 1 <= len(cache_ids) <= 50 or
                any(not isinstance(value, str) or not _CACHE_ID.fullmatch(value) for value in cache_ids)
                or len(set(cache_ids)) != len(cache_ids)):
            raise ValueError("Select one to 50 distinct generated partition caches.")
        with self._mutex:
            self._check_active()
            registry, original = self._registry()
            for cache_id in cache_ids:
                if cache_id not in registry["entries"]:
                    raise KeyError("Cache entry not found")
                reason = self._delete_reason(cache_id, registry["entries"][cache_id])
                if reason:
                    raise PermissionError(reason)
            index_path = self._path("cache/index.json")
            with ExitStack() as locks:
                locks.enter_context(_lock(self._path("cache/.index.lock", missing=True), timeout=self.lock_timeout))
                for cache_id in sorted(cache_ids):
                    locks.enter_context(_lock(self._path(f"cache/partitions/{cache_id}/.build.lock", missing=True), timeout=self.lock_timeout))
                self._check_active()
                registry, original = self._registry()
                for cache_id in cache_ids:
                    entry = registry["entries"].get(cache_id)
                    if entry is None:
                        raise KeyError("Cache entry not found")
                    reason = self._delete_reason(cache_id, entry)
                    if reason:
                        raise PermissionError(reason)
                trash = self._path("cache/.trash", missing=True)
                trash.mkdir(exist_ok=True)
                batch = trash / uuid.uuid4().hex
                batch.mkdir()
                manifest = {"schema_version": 1, "created_at": time.time(), "status": "planned", "entries": []}
                for cache_id in cache_ids:
                    for name in ("splits.json", "splits.npz"):
                        source = self._path(f"cache/partitions/{cache_id}/{name}", missing=True)
                        if source.is_file():
                            manifest["entries"].append({"cache_id": cache_id,
                                "source": source.relative_to(self.data_root).as_posix(),
                                "destination": f"{cache_id}/{name}"})
                manifest["registry_entries"] = {key: registry["entries"][key] for key in cache_ids}
                manifest_path = batch / "manifest.json"
                def publish():
                    _atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
                publish()
                moved = []
                try:
                    for entry in manifest["entries"]:
                        source = self._path(entry["source"])
                        destination = batch / entry["destination"]
                        destination.parent.mkdir(exist_ok=True)
                        source.rename(destination)
                        moved.append((source, destination))
                    for cache_id in cache_ids:
                        del registry["entries"][cache_id]
                    _atomic_bytes(index_path, (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode())
                    manifest["status"] = "deleted"
                    publish()
                except OSError:
                    failed = []
                    for source, destination in reversed(moved):
                        try:
                            destination.rename(source)
                        except OSError:
                            failed.append(source.relative_to(self.data_root).as_posix())
                    if not failed and original is not None:
                        _atomic_bytes(index_path, original)
                    manifest.update(status="recovery_required" if failed else "rolled_back", recovery_required=failed)
                    publish()
                    if failed:
                        raise RuntimeError("Some caches require recovery using " + manifest_path.relative_to(self.data_root).as_posix()) from None
                    raise
                return {"deleted_cache_ids": list(cache_ids), "count": len(cache_ids),
                        "recovery_manifest": manifest_path.relative_to(self.data_root).as_posix()}
