#!/usr/bin/env python3
"""Build disposable radar/skeleton acceleration caches from full published PKLs."""

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from .download import file_hash as sha256


INDEXES = {
    "chunk_s42": "keypoint_muti/keypoint_muti_tmax5_index.pkl",
    "chunk_s50": "keypoint_muti/keypoint_muti_tmax5_index_20260729_160657_044981.pkl",
    "chunk_s66": "keypoint_muti/keypoint_muti_tmax5_index_20260730_134840_459842.pkl",
    "chunk_s69": "keypoint_muti/keypoint_muti_tmax5_index_20260731_145352_328897.pkl",
    "chunk_s87": "keypoint_muti/keypoint_muti_tmax5_index_20260801_222359_932524.pkl",
    "subject_s42": "keypoint_muti/keypoint_muti_tmax5_index_20260803_050212_245005.pkl",
    "single_chunk_s42": "keypoint/keypoint_index_20260730_181808_758776.pkl",
    "single_subject_s42": "keypoint/keypoint_index_20260803_051001_476629.pkl",
    "single_intra_s42": "keypoint/keypoint_index.pkl",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_index(source, name, relative):
    path = source / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen paper index: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)  # Trusted local source only; release uses NPZ.
    expected_seed = int(name.rsplit("s", 1)[1])
    expected_len = 1 if name.startswith("single_") else 5
    expected_strategy = "subject" if "subject" in name else "intra_sequence" if "intra" in name else "sequence_chunk"
    for key, expected in (("seed", expected_seed), ("seq_len", expected_len),
                          ("sample_dim", 5), ("split_strategy", expected_strategy)):
        if data.get(key) != expected:
            raise ValueError(f"Frozen index {path.name}: expected {key}={expected}, got {data.get(key)}")
    for split in ("train", "val", "test"):
        if len(data["samples"][split]) != data["split_num_samples"][split]:
            raise ValueError(f"Frozen index {path.name}: corrupt {split} count")
    return path, data


def prepare_training_cache(source, output, limit_sequences=None, dry_run=False, split_names=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("Training cache must not overwrite the full dataset root")
    full_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if full_manifest.get("dataset_kind") != "full_multimodal":
        raise ValueError("Training caches must be derived from the full anonymized dataset")
    names = split_names or list(INDEXES)
    if any(name not in INDEXES for name in names):
        raise ValueError("Unknown frozen split name")
    old_manifest = output / "manifest.json"
    prior_manifest = json.loads(old_manifest.read_text(encoding="utf-8")) if old_manifest.exists() else {}
    if limit_sequences is not None and prior_manifest and not prior_manifest.get("partial", False):
        raise ValueError("Use a separate output for partial caches; a complete cache must not be truncated")
    for name in names:
        relative = INDEXES[name]
        entry = full_manifest["files"][relative]
        path = source / relative
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen index checksum failed before pickle loading: {relative}")
    indexes = {name: load_index(source, name, INDEXES[name]) for name in names}
    signature = [row for row in next(iter(indexes.values()))[1]["dataset_signature"]
                 if row[0] in full_manifest["sequences"]]
    if limit_sequences is not None and limit_sequences < 1:
        raise ValueError("--limit-sequences must be positive")
    signature = signature[:limit_sequences]
    subject_map = json.loads((source / "raw" / "manifest.json").read_text(encoding="utf-8"))["sequences"]
    entries = {}
    for sequence_id, count, filename, *_ in signature:
        cache = source / "raw" / filename
        if not cache.is_file():
            raise FileNotFoundError(f"Missing full anonymized sequence PKL: {cache}")
        if filename not in subject_map:
            raise ValueError(f"Missing subject mapping for {filename}")
        entries[sequence_id] = {"file": "raw/" + Path(filename).with_suffix(".npz").name,
                                "frames": int(count), "subject_id": subject_map[filename],
                                "source_name": filename, "source_cache": cache.name}
    print(f"Validated {len(entries)} core sequences and {len(indexes)} frozen indexes.", flush=True)
    if dry_run:
        print(f"Destination: {output}; no files written.")
        return
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir(parents=True, exist_ok=True)
    for number, (sequence_id, entry) in enumerate(entries.items(), 1):
        cache = source / "raw" / entry["source_name"]
        destination = output / entry["file"]
        source_digest = full_manifest["sequences"][sequence_id]["sha256"]
        marker = destination.with_suffix(".json")
        if destination.is_file() and marker.is_file():
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("source_sha256") == source_digest and prior.get("sha256") == sha256(destination):
                entry.update(prior)
                print(f"[{number}/{len(entries)}] verified {destination.name}", flush=True)
                continue
        if cache.stat().st_size != full_manifest["sequences"][sequence_id]["size_bytes"] or sha256(cache) != source_digest:
            raise ValueError(f"Sequence checksum failed before pickle loading: {cache}")
        with cache.open("rb") as stream:
            value = pickle.load(stream)
        joints = np.asarray(value["keypoint_frames"], dtype=np.float32)
        radar = [np.asarray(frame, dtype=np.float32) for frame in value["radar_frames"]]
        if len(radar) != entry["frames"] or joints.shape != (entry["frames"], 19, 3):
            raise ValueError(f"Cache frame/joint shape disagrees with frozen signature: {cache.name}")
        if any(frame.ndim != 2 or frame.shape[1] != 5 for frame in radar):
            raise ValueError(f"Cache must contain [N,5] radar frames: {cache.name}")
        offsets = np.concatenate(([0], np.cumsum([len(frame) for frame in radar]))).astype(np.int64)
        flat = np.concatenate(radar, axis=0)
        if destination.is_file() and marker.is_file():
            # Adopt the earlier lite-derived cache only after exact full-PKL parity.
            with np.load(destination, allow_pickle=False) as old:
                same = (np.array_equal(old["points"], flat, equal_nan=True)
                        and np.array_equal(old["joints"], joints, equal_nan=True)
                        and np.array_equal(old["offsets"], offsets))
            if same:
                entry.update(source_sha256=source_digest, sha256=sha256(destination),
                             bytes=destination.stat().st_size, points=int(len(flat)))
                write_json(marker, entry)
                print(f"[{number}/{len(entries)}] full-PKL parity verified {destination.name}", flush=True)
                continue
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, points=flat, offsets=offsets, joints=joints)
        temporary.replace(destination)
        entry.update(source_sha256=source_digest, sha256=sha256(destination),
                     bytes=destination.stat().st_size, points=int(len(flat)))
        write_json(marker, entry)
        print(f"[{number}/{len(entries)}] converted {destination.name}", flush=True)
    aliases = sorted(entries)
    lookup = {name: i for i, name in enumerate(aliases)}
    for name, (path, index) in indexes.items():
        arrays = {}
        for split in ("train", "val", "test"):
            rows = [(lookup[seq], int(start), int(end)) for seq, start, end in index["samples"][split] if seq in lookup]
            arrays[split] = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
            for seq_index, start, end in arrays[split]:
                if start < 0 or end > entries[aliases[seq_index]]["frames"] or end - start != index["seq_len"]:
                    raise ValueError(f"Invalid frozen {name}/{split} window")
        npz_path = output / "splits" / (name + ".npz")
        np.savez_compressed(npz_path, **arrays)
        metadata = {key: index[key] for key in ("seed", "seq_len", "sample_dim", "partition",
                    "split_strategy", "split_chunk_size", "max_timestamp_gap_ms", "min_valid_points", "feature_stats") if key in index}
        metadata.update(schema_version=1, sequence_ids=aliases, source_index=path.name,
                        source_index_sha256=sha256(path), index_sha256=sha256(npz_path),
                        source_split_num_samples=index["split_num_samples"],
                        split_num_samples={key: len(value) for key, value in arrays.items()},
                        validity="Imported frozen windows; original timestamp continuity was validated by the source index.",
                        partial=limit_sequences is not None)
        if "split_subject_ids" in index:
            metadata["split_subject_ids"] = index["split_subject_ids"]
        write_json(npz_path.with_suffix(".json"), metadata)
        print(f"Split {name}: {metadata['split_num_samples']}", flush=True)
    same_coverage = set(prior_manifest.get("sequences", {})) == set(entries)
    prior_splits = prior_manifest.get("splits", []) if same_coverage else []
    manifest = {"schema_version": 1, "dataset_kind": "training_cache", "sequences": entries,
                "splits": list(dict.fromkeys([*prior_splits, *indexes])),
                "partial": limit_sequences is not None or full_manifest.get("partial", False),
                "source_manifest_sha256": sha256(source / "manifest.json"),
                "modalities": ["radar", "skeleton"],
                "timestamps_available": False,
                "preprocessing": "Derived from anonymized full PKLs; source timestamps remain in original PKLs; frozen splits preserved."}
    write_json(output / "manifest.json", manifest)
    print(f"Prepared training cache: {output}; {len(entries)} sequences.", flush=True)


def training_root(root, split_name):
    """Resolve full data versus numeric fixtures; build missing acceleration files."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_kind") != "full_multimodal":
        return root  # Numeric example fixtures and explicitly addressed caches.
    cache = root / "cache/pose-v1"
    name = Path(split_name).stem
    cached = json.loads((cache / "manifest.json").read_text(encoding="utf-8")) if (cache / "manifest.json").is_file() else {}
    ready = cached.get("source_manifest_sha256") == sha256(root / "manifest.json")
    ready = ready and set(cached.get("sequences", {})) == set(manifest["sequences"])
    metadata_path = cache / "splits" / (name + ".json")
    index_path = metadata_path.with_suffix(".npz")
    if ready and metadata_path.is_file() and index_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ready = (metadata["index_sha256"] == sha256(index_path)
                 and set(metadata["sequence_ids"]).issubset(cached["sequences"]))
    else:
        ready = False
    if not ready:
        print("Building radar/skeleton training cache from trusted full PKLs; images remain in raw/.", flush=True)
        prepare_training_cache(root, cache, split_names=[name])
    return cache


def record_config(root, cache, config, metadata):
    """Bind a shared numeric cache to the exact runtime preprocessing settings."""
    root, cache = Path(root), Path(cache)
    if root.resolve() == cache.resolve():
        return None
    descriptor = {"format_version": 1, "source_manifest_sha256": sha256(root / "manifest.json"),
                  "frozen_index_sha256": metadata["index_sha256"], "config": config,
                  "feature_stats": metadata["feature_stats"],
                  "sampling": "Runtime NumPy random-start FPS; never frozen across overlapping windows"}
    identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    directory = cache / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (identity + ".json")
    if not path.exists():
        write_json(path, descriptor)
    return identity
