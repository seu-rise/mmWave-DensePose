"""PCC-only spherical supervision and frozen completion splits.

Labels are generated online with the original Torch random sampling. Shared
numeric radar/skeleton arrays remain separate from these task-specific indexes.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import training_root, write_json
from .completion_indexes import COMPLETION_INDEXES, COMPLETION_INDEX_SHA256
from .dataset import align_points, normalize_points
from .download import file_hash


def generate_pcc_label(joints, radar, radius=0.05):
    """Original 16 uniform sphere-surface samples per joint, followed by radar."""
    if joints.shape != (19, 3) or radar.ndim != 2 or radar.shape[1] != 3 or len(radar) == 0:
        raise ValueError("PCC labels require [19,3] joints and nonempty [N,3] radar")
    if not torch.isfinite(joints).all() or not torch.isfinite(radar).all():
        raise ValueError("PCC joints and radar must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("PCC label radius must be positive and finite")
    # Keep source shapes, draw order and arithmetic for seeded parity.
    theta = 2 * np.pi * torch.rand(1, 19, 16, device=joints.device)
    phi = torch.arccos(2 * torch.rand(1, 19, 16, device=joints.device) - 1)
    xyz = torch.stack((radius * torch.sin(phi) * torch.cos(theta),
                       radius * torch.sin(phi) * torch.sin(theta), radius * torch.cos(phi)), dim=-1)
    sphere = (xyz + joints[None, :, None]).reshape(304, 3)
    return torch.cat((sphere, radar), dim=0)


def align_supervision_radar(points, num_points):
    """Remove input padding; repeat real points rather than supervising zeros."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PCC supervision radar must have shape [N,3]")
    points = points[np.any(np.abs(points) > 1e-8, axis=1)]
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(points) < num_points:
        return points[np.resize(np.arange(len(points)), num_points)]
    return align_points(points, num_points)


def _numeric_root(data_root):
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    numeric = training_root(root, "chunk_s42") if manifest.get("dataset_kind") == "full_multimodal" else root
    if not (numeric / "manifest.json").is_file():
        raise FileNotFoundError("Missing shared numeric cache; use a complete installed dataset and retry")
    return numeric, manifest


def numeric_sequences_signature(manifest):
    """Bind shared sequence identities, independent of other tasks' split exports."""
    descriptor = {"sequences": manifest["sequences"], "partial": bool(manifest.get("partial", False))}
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_completion_cache(source, data_root, trust_pickle=False, dry_run=False):
    """Copy exact frozen PCC indexes and export safe numeric split arrays.

    Source may be the original dataset or an installed release containing dense/.
    Reading indexes executes pickle, so the caller must explicitly trust source.
    No image PKL or radar/skeleton array is rewritten or decoded by this step.
    """
    if not trust_pickle:
        raise ValueError("Completion index preparation requires --trust-pickle for a trusted source")
    source, root = Path(source), Path(data_root)
    numeric, full_manifest = _numeric_root(root)
    numeric_manifest_path = numeric / "manifest.json"
    numeric_manifest = json.loads(numeric_manifest_path.read_text(encoding="utf-8"))
    original_source_digest = file_hash(root / "manifest.json")
    entries = numeric_manifest["sequences"]
    aliases = sorted(entries)
    lookup = {name: index for index, name in enumerate(aliases)}
    cache = root / "cache/completion-v1"
    exports = {}
    for name, relative in COMPLETION_INDEXES.items():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen PCC index: {path}")
        digest = file_hash(path)
        if not full_manifest.get("synthetic", False) and digest != COMPLETION_INDEX_SHA256[name]:
            raise ValueError(f"Frozen PCC index differs from the paper training log: {relative}")
        recorded = full_manifest.get("files", {}).get(relative)
        if recorded and (recorded["sha256"] != digest or recorded["size_bytes"] != path.stat().st_size):
            raise ValueError(f"Frozen PCC index checksum mismatch: {relative}")
        with path.open("rb") as stream:
            index = pickle.load(stream)  # Explicitly trusted, exact source filename.
        strategy = "subject" if "subject" in name else "sequence"
        for key, expected in (("task", "dense"), ("seed", 42), ("seq_len", 1),
                              ("sample_dim", 3), ("split_strategy", strategy)):
            if index.get(key) != expected:
                raise ValueError(f"Frozen PCC {name} requires {key}={expected}")
        signature = {row[0]: row for row in index["dataset_signature"]}
        if not set(entries).issubset(signature):
            raise ValueError(f"Shared numeric sequences do not match PCC signature: {name}")
        for sequence, entry in entries.items():
            row = signature[sequence]
            if int(row[1]) != int(entry["frames"]):
                raise ValueError(f"PCC signature frame count mismatch: {sequence}")
            expected_name = entry.get("source_name", Path(entry["file"]).with_suffix(".pkl").name)
            if row[2] != expected_name:
                raise ValueError(f"PCC signature filename mismatch: {sequence}")
            if not (numeric / entry["file"]).is_file():
                raise FileNotFoundError(f"Missing shared numeric sequence: {entry['file']}")
        partial = bool(full_manifest.get("partial") or numeric_manifest.get("partial")
                       or set(entries) != set(signature))
        arrays = {}
        for split in ("train", "val", "test"):
            samples = index["samples"][split]
            if len(samples) != index["split_num_samples"][split]:
                raise ValueError(f"Frozen PCC split count mismatch: {name}/{split}")
            rows = []
            for sequence, start, end in samples:
                if sequence not in signature:
                    raise ValueError(f"Unknown sequence in frozen PCC index: {sequence}")
                if start < 0 or end != start + 1 or end > signature[sequence][1]:
                    raise ValueError(f"Invalid frozen PCC frame: {name}/{split}")
                if sequence in lookup:
                    rows.append((lookup[sequence], start, end))
            arrays[split] = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
        metadata = {key: index[key] for key in ("seed", "seq_len", "sample_dim", "partition",
                    "split_strategy", "split_chunk_size", "max_timestamp_gap_ms", "min_valid_points",
                    "feature_stats", "split_subject_ids") if key in index}
        metadata.update(schema_version=1, task="completion", model="PCCCompletion", sequence_ids=aliases,
                        source_index=relative, source_index_sha256=digest,
                        source_split_num_samples=index["split_num_samples"],
                        split_num_samples={key: len(value) for key, value in arrays.items()},
                        partial=partial, synthetic=bool(full_manifest.get("synthetic", False)),
                        numeric_manifest_sha256=file_hash(numeric_manifest_path),
                        validity="Exact source window order and original validity decisions retained")
        exports[name] = {"metadata": metadata, "arrays": arrays, "source": path}
    if dry_run:
        return {name: item["metadata"]["split_num_samples"] for name, item in exports.items()}
    # Check both destinations before any copy or manifest update.
    for name, item in exports.items():
        destination = root / COMPLETION_INDEXES[name]
        if destination.exists() and file_hash(destination) != item["metadata"]["source_index_sha256"]:
            raise FileExistsError(f"Refusing to overwrite different frozen index: {destination}")
    for name, item in exports.items():
        destination = root / COMPLETION_INDEXES[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(item["source"], destination)
        if (destination.stat().st_size != item["source"].stat().st_size
                or file_hash(destination) != item["metadata"]["source_index_sha256"]):
            raise ValueError(f"Copied PCC index checksum mismatch: {destination}")
    if full_manifest.get("dataset_kind") == "full_multimodal":
        updated = dict(full_manifest)
        updated["completion_indexes"] = COMPLETION_INDEXES
        updated["completion_training_cache"] = "cache/completion-v1"
        updated["files"] = dict(full_manifest.get("files", {}))
        for name, item in exports.items():
            relative = COMPLETION_INDEXES[name]
            updated["files"][relative] = {"sha256": item["metadata"]["source_index_sha256"],
                                           "size_bytes": (root / relative).stat().st_size}
        if updated != full_manifest:
            write_json(root / "manifest.json", updated)
    source_digest = file_hash(root / "manifest.json")
    # Adding index metadata does not change sequence arrays. Advance the shared
    # cache's source binding only when every original PKL identity already agrees.
    # Legacy lite-derived caches do not satisfy this condition and are not adopted.
    source_sequences = full_manifest.get("sequences", {})
    can_advance = (source_digest != original_source_digest
                   and numeric_manifest.get("source_manifest_sha256") == original_source_digest
                   and set(entries) == set(source_sequences))
    if can_advance:
        can_advance = all(
            entry.get("source_sha256") == source_sequences[key].get("sha256")
            and entry.get("frames") == source_sequences[key].get("frames")
            and entry.get("source_name") == Path(source_sequences[key]["file"]).name
            for key, entry in entries.items()
        )
    if can_advance:
        numeric_manifest["source_manifest_sha256"] = source_digest
        write_json(numeric_manifest_path, numeric_manifest)
    numeric_digest = file_hash(numeric_manifest_path)
    numeric_signature = numeric_sequences_signature(numeric_manifest)
    for name, item in exports.items():
        item["metadata"]["source_manifest_sha256"] = source_digest
        item["metadata"]["numeric_manifest_sha256"] = numeric_digest
        item["metadata"]["numeric_sequences_sha256"] = numeric_signature
        index_path = cache / "splits" / (name + ".npz")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **item["arrays"])
        temporary.replace(index_path)
        item["metadata"]["index_sha256"] = file_hash(index_path)
        write_json(index_path.with_suffix(".json"), item["metadata"])
    manifest = {"schema_version": 1, "dataset_kind": "completion_cache", "model": "PCCCompletion",
                "splits": list(exports), "shared_numeric_root": str(numeric.relative_to(root)),
                "source_manifest_sha256": file_hash(root / "manifest.json"),
                "numeric_manifest_sha256": file_hash(numeric_manifest_path),
                "numeric_sequences_sha256": numeric_signature,
                "partial": any(item["metadata"]["partial"] for item in exports.values()),
                "label_policy": "Online 16 sphere-surface points per joint plus repeated real radar; no frozen labels"}
    write_json(cache / "manifest.json", manifest)
    return {name: item["metadata"]["split_num_samples"] for name, item in exports.items()}


def _completion_cache_files_exist(root, split_name):
    root = Path(root)
    cache = root / "cache/completion-v1"
    name = Path(split_name).stem
    metadata_path = cache / "splits" / (name + ".json")
    index_path = metadata_path.with_suffix(".npz")
    return all(path.is_file() for path in
               (cache / "manifest.json", root / "cache/pose-v1/manifest.json", metadata_path, index_path))


def ensure_completion_cache(data_root, split_name):
    """Build the requested PCC split cache lazily from a complete installed release."""
    root = Path(data_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_kind") != "full_multimodal":
        return
    split = Path(split_name)
    if split.is_file():
        return
    name = split.stem
    if name not in COMPLETION_INDEXES:
        raise ValueError(f"Unknown PCC completion split: {name}")
    # Resolve the shared numeric arrays before preparing the PCC-specific index.
    numeric, full = _numeric_root(root)
    ready = _completion_cache_files_exist(root, name)
    if ready:
        metadata_path = root / "cache/completion-v1/splits" / (name + ".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            numeric_manifest = json.loads((numeric / "manifest.json").read_text(encoding="utf-8"))
            signature = numeric_sequences_signature(numeric_manifest)
            ready = (metadata.get("source_manifest_sha256") == file_hash(root / "manifest.json")
                     and metadata.get("numeric_sequences_sha256") == signature
                     and metadata.get("index_sha256") == file_hash(metadata_path.with_suffix(".npz")))
            if metadata.get("numeric_sequences_sha256") != signature:
                # Re-export after a legitimate shared-cache rebuild, but never
                # bless an arbitrary change to the stored sequence provenance.
                for key, entry in numeric_manifest["sequences"].items():
                    original = full["sequences"].get(key, {})
                    if (entry.get("source_sha256") != original.get("sha256")
                            or entry.get("frames") != original.get("frames")):
                        raise ValueError("PCC split belongs to different numeric sequences; source identity mismatch")
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            ready = False
    if not ready:
        prepare_completion_cache(root, root, trust_pickle=True)


class CompletionDataset(Dataset):
    """PCC-only XYZ single-frame input and online spherical dense supervision."""

    def __init__(self, data_root, split_manifest="pcc_sequence_s42", split="train", num_points=80,
                 input_dim=3, seq_len=1, cache_size=2, seed=42, pcc_label_radius=0.05, prepared=None):
        if input_dim != 3 or seq_len != 1:
            raise ValueError("PCCCompletion requires input_dim=3 and seq_len=1")
        if split not in ("train", "val", "test") or num_points < 1 or cache_size < 1:
            raise ValueError("Require train/val/test and positive point/cache sizes")
        if not np.isfinite(pcc_label_radius) or pcc_label_radius <= 0:
            raise ValueError("PCC label radius must be positive and finite")
        self.source_root = Path(data_root)
        if prepared:
            self.data_root = Path(prepared["numeric_root"])
            source_manifest = json.loads((self.source_root / "manifest.json").read_text(encoding="utf-8"))
        else:
            ensure_completion_cache(self.source_root, split_manifest)
            self.data_root, source_manifest = _numeric_root(self.source_root)
        manifest_path = self.data_root / "manifest.json"
        numeric_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = Path(prepared["metadata_path"]) if prepared else Path(split_manifest)
        if not path.is_file():
            path = self.source_root / "cache/completion-v1/splits" / (str(split_manifest).removesuffix(".json") + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"Missing PCC split {path}; verify the installed dataset and split configuration")
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        if self.metadata.get("task") != "completion" or self.metadata.get("model") != "PCCCompletion":
            raise ValueError("PCCCompletion must use its own frozen completion split")
        if self.metadata.get("seq_len") != 1 or self.metadata.get("sample_dim") != 3:
            raise ValueError("PCC split must contain single-frame XYZ samples")
        if self.metadata.get("seed") != seed:
            raise ValueError("PCC seed must match the frozen split")
        if self.metadata.get("source_manifest_sha256") != file_hash(self.source_root / "manifest.json"):
            raise ValueError("PCC split belongs to a different source manifest; regenerate completion cache")
        numeric_signature = numeric_sequences_signature(numeric_manifest)
        if self.metadata.get("numeric_sequences_sha256") != numeric_signature:
            raise ValueError("PCC split belongs to different numeric sequences; regenerate completion cache")
        if file_hash(path.with_suffix(".npz")) != self.metadata["index_sha256"]:
            raise ValueError("PCC numeric split checksum mismatch")
        self.metadata = dict(self.metadata, partial=bool(self.metadata.get("partial")
                                  or source_manifest.get("partial") or numeric_manifest.get("partial")))
        self.sequences = numeric_manifest["sequences"]
        self.sequence_ids = self.metadata["sequence_ids"]
        if not set(self.sequence_ids).issubset(self.sequences):
            raise ValueError("PCC split references missing numeric sequences")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            self.samples = data[split].copy()
        if self.samples.ndim != 2 or self.samples.shape[1] != 3 or self.samples.dtype.kind not in "iu":
            raise ValueError("PCC split must contain integer [N,3] windows")
        if len(self.samples) != self.metadata["split_num_samples"][split]:
            raise ValueError("PCC sample count does not match metadata")
        if len(self.samples):
            counts = np.asarray([self.sequences[key]["frames"] for key in self.sequence_ids])
            ids, starts, ends = self.samples.T
            if (ids < 0).any() or (ids >= len(counts)).any():
                raise ValueError("PCC split contains invalid sequence references")
            if (starts < 0).any() or (ends != starts + 1).any() or (ends > counts[ids]).any():
                raise ValueError("PCC split contains invalid single-frame windows")
        self.feature_stats = self.metadata.get("feature_stats", {})
        self.num_points, self.seq_len, self.input_dim = num_points, 1, 3
        self.cache_size, self.split, self.radius = cache_size, split, float(pcc_label_radius)
        self._cache = OrderedDict()
        identity = {"task": "completion", "model": "PCCCompletion", "index_sha256": self.metadata["index_sha256"],
                    "numeric_sequences_sha256": numeric_signature, "num_points": num_points,
                    "radius": self.radius, "samples_per_joint": 16, "seed": seed,
                    "sampling": "Runtime NumPy FPS and Torch sphere sampling"}
        self.cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if prepared:
            self.cache_identity = prepared["cache_id"]

    def __len__(self):
        return len(self.samples)

    def _sequence(self, key):
        if key not in self._cache:
            with np.load(self.data_root / self.sequences[key]["file"], allow_pickle=False) as data:
                self._cache[key] = tuple(data[field].copy() for field in ("points", "offsets", "joints"))
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __getitem__(self, index):
        sequence_index, start, end = map(int, self.samples[index])
        key = self.sequence_ids[sequence_index]
        points, offsets, joints = self._sequence(key)
        xyz = align_points(normalize_points(points[offsets[start]:offsets[end]], 3, self.feature_stats), self.num_points)
        joint = torch.from_numpy(joints[start].copy())
        radar = torch.from_numpy(align_supervision_radar(xyz, self.num_points))
        target = generate_pcc_label(joint, radar, self.radius)
        return {"points": torch.from_numpy(xyz[None]), "target": target, "sample_id": f"{key}:{start}:{end}"}
