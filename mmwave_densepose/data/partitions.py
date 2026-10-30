"""Model/strategy/seed driven partitions and one index for disposable caches.

Chunks are assigned before windows, subjects never cross splits, statistics use
only unique training frames, and the seed is part of both partitioning and stats.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
import uuid

import numpy as np

from .download import file_hash
from .installation_state import ensure_complete_dataset

SPLITS = ("train", "val", "test")
STRATEGIES = ("sequence_chunk", "subject")
SINGLE_MODELS = {"DGCNN", "PointTransformer", "PointMLP", "PoseGraphNet", "PointNet2Baseline"}
TEMPORAL_MODELS = {"PASTNet", "MMPoseFK", "DualPathCNNBiLSTM", "PntPoseAT", "GlobalLocalNetwork"}
ALGORITHM_VERSION = 2


def _integer(value, name, minimum=1):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def normalize_data_config(config):
    """Resolve model capabilities and public split options without reading files."""
    result = dict(config)
    if "split_manifest" in result:
        raise ValueError("split_manifest is no longer a split option; select split_strategy=sequence_chunk or subject")
    model = result.get("model")
    if model not in SINGLE_MODELS | TEMPORAL_MODELS | {"PCCCompletion"}:
        raise ValueError(f"Unknown data model: {model}")
    completion = model == "PCCCompletion"
    family = "completion" if completion else "single_frame" if model in SINGLE_MODELS else "temporal"
    task = "completion" if completion else "pose"
    if result.get("task", task) != task:
        raise ValueError(f"{model} requires task={task}")
    result.update(task=task, data_family=family)
    length = _integer(result.get("seq_len", 1 if family != "temporal" else 5), "seq_len")
    if family != "temporal" and length != 1:
        raise ValueError(f"{model} uses single-frame data (seq_len=1)")
    result["seq_len"] = length
    strategy = result.setdefault("split_strategy", "sequence_chunk")
    if strategy not in STRATEGIES:
        raise ValueError("split_strategy must be sequence_chunk or subject")
    result["seed"] = _integer(result.get("seed", 42), "seed", 0)
    chunk = result.get("split_chunk_size", 200 if strategy == "sequence_chunk" else None)
    result["split_chunk_size"] = None if strategy == "subject" and chunk is None else _integer(chunk, "split_chunk_size")
    result["min_valid_points"] = _integer(result.get("min_valid_points", 50), "min_valid_points", 0)
    gap = float(result.get("max_timestamp_gap_ms", 300.0))
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("max_timestamp_gap_ms must be positive and finite")
    result["max_timestamp_gap_ms"] = gap
    partition = np.asarray(result.get("partition", [.8, .1, .1]), dtype=np.float64)
    if partition.shape != (3,) or not np.isfinite(partition).all() or (partition < 0).any() or partition.sum() <= 0:
        raise ValueError("partition requires three nonnegative finite ratios with a positive sum")
    result["partition"] = (partition / partition.sum()).tolist()
    result["num_points"] = _integer(result.get("num_points", 80 if completion else 300), "num_points")
    result["input_dim"] = _integer(result.get("input_dim", 3 if completion else 5), "input_dim")
    if result["input_dim"] not in (3, 4, 5) or completion and result["input_dim"] != 3:
        raise ValueError("input_dim must be 3/4/5; PCCCompletion requires XYZ (3)")
    result["num_joints"] = _integer(result.get("num_joints", 19), "num_joints")
    if completion:
        result["num_out_points"] = _integer(result.get("num_out_points", 196), "num_out_points")
        radius = float(result.get("pcc_label_radius", .05))
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("pcc_label_radius must be positive and finite")
        result["pcc_label_radius"] = radius
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mkdir(path):
    # Windows may briefly deny concurrent creation of the same ancestor.
    for attempt in range(20):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return
        except (FileNotFoundError, PermissionError):
            if attempt == 19:
                raise
            time.sleep(.01)


def _json(path, value):
    path = Path(path)
    _mkdir(path.parent)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz(path, **arrays):
    path = Path(path)
    _mkdir(path.parent)
    temporary = path.with_name("." + path.stem + "." + uuid.uuid4().hex + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path, timeout=1800):
    """OS-owned locks release automatically when a process exits or crashes."""
    path = Path(path)
    _mkdir(path.parent)
    deadline = time.monotonic() + timeout
    stream = path.open("a+b")
    if os.name == "nt":
        import msvcrt
        if path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        def acquire():
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        def release():
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        def acquire(): fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        def release(): fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    try:
        while True:
            try:
                acquire(); break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for active cache build: {path.name}")
                time.sleep(.05)
    except BaseException:
        stream.close()
        raise
    try:
        yield
    finally:
        release()
        stream.close()


def _safe(root, relative):
    path = Path(root) / relative
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset/cache paths must remain inside the dataset root")
    return resolved


def _source(root):
    root = Path(root).resolve()
    ensure_complete_dataset(root)
    path = root / "manifest.json"
    manifest = _read(path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sequences"), dict) or not manifest["sequences"]:
        raise ValueError("Dataset manifest must declare its sequence inventory")
    inventory = []
    for sequence, item in sorted(manifest["sequences"].items()):
        file = _safe(root, item["file"])
        if not file.is_file():
            raise FileNotFoundError(f"Missing dataset sequence: {file.name}")
        stat = file.stat()
        if "size_bytes" in item and stat.st_size != item["size_bytes"]:
            raise ValueError(f"Source sequence size mismatch: {file.name}")
        inventory.append([sequence, item["file"], item.get("sha256"), stat.st_size, stat.st_mtime_ns])
    return manifest, file_hash(path), _digest(inventory)


def _parameters(config):
    keys = ("model", "task", "data_family", "split_strategy", "seed", "seq_len", "num_points",
            "input_dim", "num_joints", "partition", "split_chunk_size", "max_timestamp_gap_ms",
            "min_valid_points")
    if config["model"] == "PCCCompletion":
        keys += ("num_out_points", "pcc_label_radius")
    return {key: config[key] for key in keys if key in config}


def _descriptor(root, config):
    _safe(root, "cache")
    manifest, source_hash, inventory_hash = _source(root)
    parameters = _parameters(normalize_data_config(config))
    descriptor = {"algorithm_version": ALGORITHM_VERSION, "source_manifest_sha256": source_hash,
                  "source_inventory_sha256": inventory_hash,
                  "parameters": parameters}
    cache_id = _digest(descriptor)
    full = manifest.get("dataset_kind") == "full_multimodal"
    return manifest, descriptor, cache_id, _safe(root, "cache/sequences-v2") if full else Path(root)


def _legacy_entries(root):
    """Inventory old derivatives without opening raw PKLs or sequence arrays."""
    found = {}
    for name in ("pose-v1", "completion-v1"):
        directory = root / "cache" / name
        if not directory.is_dir():
            continue
        paths = [directory / "manifest.json", *sorted((directory / "splits").glob("*.json")),
                 *sorted((directory / "configs").glob("*.json"))]
        for path in paths:
            try:
                resolved = _safe(root, path.relative_to(root))
                if not resolved.is_file(): continue
                document = _read(resolved)
                if not isinstance(document, dict): continue
                relative = resolved.relative_to(root).as_posix()
                kind = "manifest" if path.name == "manifest.json" else "configuration" if path.parent.name == "configs" else "partition"
                item = {"kind": "legacy_" + kind, "compatible_with_generated_partitions": False,
                        "metadata_path": relative, "metadata_sha256": file_hash(resolved),
                        "original_provenance": {key: document[key] for key in
                           ("source_manifest_sha256", "source_index", "source_index_sha256", "seed", "seq_len",
                            "sample_dim", "split_strategy", "split_num_samples", "config", "frozen_index_sha256") if key in document}}
                if kind == "partition" and path.with_suffix(".npz").is_file():
                    index = _safe(root, path.with_suffix(".npz").relative_to(root))
                    item.update(index_path=index.relative_to(root).as_posix(), index_sha256=file_hash(index),
                                declared_index_sha256=document.get("index_sha256"))
                if kind == "manifest":
                    item["sequence_files"] = [{"path": (directory / value["file"]).relative_to(root).as_posix(),
                         "recorded_sha256": value.get("sha256"), "recorded_source_sha256": value.get("source_sha256"),
                         "recorded_bytes": value.get("bytes", value.get("size_bytes"))}
                         for value in document.get("sequences", {}).values()]
                found["legacy_" + _digest(relative)] = item
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return found


def _orphan_partitions(root, known):
    """Make unregistered files discoverable without blessing their provenance."""
    entries = {}
    for path in sorted((root / "cache/partitions").glob("*/splits.json")):
        cache_id = path.parent.name
        if cache_id in known:
            continue
        try:
            path = _safe(root, path.relative_to(root))
            metadata = _read(path)
            if not isinstance(metadata, dict): continue
            entries[cache_id] = {"kind": "orphan_partition", "compatible_with_generated_partitions": False,
                                 "metadata_path": path.relative_to(root).as_posix(),
                                 "metadata_sha256": file_hash(path),
                                 "original_provenance": {key: metadata.get(key) for key in
                                     ("model", "split_strategy", "seed", "source_manifest_sha256", "split_num_samples")}}
        except (OSError, ValueError, TypeError):
            continue
    return entries


def index_existing_caches(data_root):
    """Register legacy caches for discovery only, without generating training data."""
    root = Path(data_root).resolve()
    _safe(root, "cache")
    with _lock(root / "cache/.index.lock"):
        path = _safe(root, "cache/index.json")
        try: registry = _read(path)
        except (OSError, ValueError): registry = {"schema_version": 2, "entries": {}}
        if not isinstance(registry, dict) or not isinstance(registry.get("entries"), dict):
            registry = {"schema_version": 2, "entries": {}}
        original = json.dumps(registry, sort_keys=True)
        entries = registry.setdefault("entries", {})
        entries.update(_legacy_entries(root))
        entries.update(_orphan_partitions(root, entries))
        registry["legacy_roots"] = [p.relative_to(root).as_posix() for p in
            (root / "cache/pose-v1", root / "cache/completion-v1") if p.is_dir()]
        if json.dumps(registry, sort_keys=True) != original or not path.exists():
            _json(path, registry)
        return registry


def _register(root, cache_id, entry, shared=None):
    root = Path(root)
    registry_path = _safe(root, "cache/index.json")
    with _lock(root / "cache/.index.lock"):
        try:
            registry = _read(registry_path)
        except (OSError, ValueError):
            registry = {"schema_version": 2, "entries": {}}
        if not isinstance(registry, dict) or not isinstance(registry.get("entries"), dict):
            registry = {"schema_version": 2, "entries": {}}
        old = json.dumps(registry, sort_keys=True)
        registry["entries"][cache_id] = entry
        if "legacy_roots" not in registry:
            registry["entries"].update(_legacy_entries(root))
            registry["entries"].update(_orphan_partitions(root, registry["entries"]))
        if shared is not None:
            registry["shared_sequences"] = shared
        # The previous derived-cache roots remain visible, but are not treated as
        # compatible generated partitions or silently rebound to a new strategy.
        registry["legacy_roots"] = [p.relative_to(root).as_posix() for p in
            (root / "cache/pose-v1", root / "cache/completion-v1") if p.is_dir()]
        if json.dumps(registry, sort_keys=True) != old or not registry_path.exists():
            _json(registry_path, registry)


def normalize_timestamp_ms(value):
    timestamps = np.asarray(value, dtype=np.float64).reshape(-1)
    timestamps = np.nan_to_num(timestamps, nan=np.nan, posinf=np.nan, neginf=np.nan)
    gaps = np.abs(np.diff(timestamps))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if not len(gaps):
        return timestamps
    median = float(np.median(gaps))
    scale = 1e6 if median > 1e7 else 1e4 if median > 1e5 else 1e3 if median > 1e3 else 1e-3 if median < 1 else 1
    return timestamps / scale


def _timestamps(value):
    for key in ("radar_timestamp", "kinect_timestamp"):
        if value.get(key) is None:
            continue
        try:
            result = np.asarray(value.get(key, []), dtype=np.float64).reshape(-1)
            if result.size:
                return normalize_timestamp_ms(result)
        except (ValueError, TypeError):
            pass
    return np.empty(0, dtype=np.float64)


def _numeric_sequences(root, manifest, source_hash, numeric, progress=None):
    """Cache original unsampled arrays once, including real time continuity data."""
    root, numeric = Path(root), Path(numeric)
    if manifest.get("dataset_kind") != "full_multimodal":
        for item in manifest["sequences"].values():
            path = _safe(root, item["file"])
            if item.get("sha256") and file_hash(path) != item["sha256"]:
                raise ValueError(f"Numeric source sequence checksum mismatch: {path.name}")
        return manifest
    _mkdir(numeric)
    with _lock(numeric / ".build.lock"):
        try:
            prior = _read(numeric / "manifest.json")
        except (OSError, ValueError):
            prior = {}
        if not isinstance(prior, dict) or not isinstance(prior.get("sequences", {}), dict):
            prior = {}
        entries = {}
        for number, (sequence, original) in enumerate(sorted(manifest["sequences"].items()), 1):
            if progress:
                progress(phase="preparing-cache", progress={"kind": "sequence", "current": number - 1,
                         "total": len(manifest["sequences"]), "label": "Preparing numeric sequences"})
            source = _safe(root, original["file"])
            source_stat = source.stat()
            destination = numeric / "raw" / (sequence + ".npz")
            previous = prior.get("sequences", {}).get(sequence, {})
            if not isinstance(previous, dict): previous = {}
            if (previous.get("source_sha256") == original.get("sha256")
                    and previous.get("frames") == original.get("frames") and destination.is_file()):
                source_unchanged = (previous.get("verified_source_size") == source_stat.st_size
                                    and previous.get("verified_source_mtime_ns") == source_stat.st_mtime_ns)
                if not source_unchanged and file_hash(source) != original.get("sha256"):
                    raise ValueError(f"Source sequence checksum mismatch before cache reuse: {source.name}")
                stat = destination.stat()
                unchanged = (previous.get("verified_size") == stat.st_size
                             and previous.get("verified_mtime_ns") == stat.st_mtime_ns)
                if unchanged or previous.get("sha256") == file_hash(destination):
                    entries[sequence] = dict(previous, subject_id=original.get("subject_id"),
                                             verified_size=stat.st_size, verified_mtime_ns=stat.st_mtime_ns,
                                             verified_source_size=source_stat.st_size,
                                             verified_source_mtime_ns=source_stat.st_mtime_ns)
                    continue
            if not original.get("sha256") or file_hash(source) != original["sha256"]:
                raise ValueError(f"Source sequence checksum mismatch before pickle loading: {source.name}")
            print(f"Preparing numeric sequence {number}/{len(manifest['sequences'])}: {source.name}", flush=True)
            with source.open("rb") as stream:
                value = pickle.load(stream)  # Trusted installed release, authenticated above.
            frames = [np.asarray(frame, dtype=np.float32) for frame in value["radar_frames"]]
            joints = np.asarray(value["keypoint_frames"], dtype=np.float32)
            if len(frames) != original["frames"] or joints.shape != (len(frames), 19, 3):
                raise ValueError(f"Sequence frame/joint shape mismatch: {source.name}")
            if any(frame.ndim != 2 or frame.shape[1] < 3 for frame in frames):
                raise ValueError(f"Invalid radar frame shape: {source.name}")
            offsets = np.concatenate(([0], np.cumsum([len(frame) for frame in frames]))).astype(np.int64)
            points = np.concatenate(frames)
            _npz(destination, points=points, offsets=offsets, joints=joints, timestamps=_timestamps(value))
            entries[sequence] = {"file": destination.relative_to(numeric).as_posix(), "frames": len(frames),
                                 "subject_id": original.get("subject_id"), "source_name": source.name,
                                 "source_sha256": original["sha256"], "sha256": file_hash(destination),
                                 "size_bytes": destination.stat().st_size,
                                 "verified_size": destination.stat().st_size,
                                 "verified_mtime_ns": destination.stat().st_mtime_ns,
                                 "verified_source_size": source_stat.st_size,
                                 "verified_source_mtime_ns": source_stat.st_mtime_ns}
        if progress:
            progress(phase="preparing-cache", progress={"kind": "sequence", "current": len(entries),
                     "total": len(entries), "label": "Numeric sequences ready"})
        result = {"schema_version": 2, "dataset_kind": "training_cache", "sequences": entries,
                  "source_manifest_sha256": source_hash, "partial": bool(manifest.get("partial", False)),
                  "timestamps_available": True}
        if result != prior:
            _json(numeric / "manifest.json", result)
        return result


def _allocate(total, ratios):
    counts = np.zeros(3, dtype=np.int64)
    active = [i for i, r in enumerate(ratios) if r > 0]
    if total >= len(active):
        counts[active] = 1
        remaining = total - len(active)
    else:
        counts[sorted(active, key=lambda i: (-ratios[i], i))[:total]] = 1
        remaining = 0
    if remaining:
        raw = np.asarray(ratios)[active] * remaining
        floor = np.floor(raw).astype(np.int64)
        counts[active] += floor
        for i in np.argsort(-(raw - floor))[:remaining - int(floor.sum())]:
            counts[active[int(i)]] += 1
    return counts


def _assign(items, ratios, rng):
    counts = _allocate(len(items), ratios)
    indices = np.arange(len(items), dtype=np.int64)
    rng.shuffle(indices)
    result, cursor = {}, 0
    for name, count in zip(SPLITS, counts):
        result[name] = [items[int(i)] for i in indices[cursor:cursor + count]]
        cursor += int(count)
    return result


def _weighted_subjects(entries, config, rng):
    subjects = {}
    for sequence, item in sorted(entries.items()):
        subject = item.get("subject_id")
        if not subject:
            raise ValueError(f"Missing subject_id for subject strategy: {sequence}")
        subjects.setdefault(subject, []).append(sequence)
    chunk, length = config["split_chunk_size"], config["seq_len"]
    if chunk is None:
        assigned = _assign(sorted(subjects), config["partition"], rng)
        sequence_split = {sequence: split for split, names in assigned.items()
                          for subject in names for sequence in subjects[subject]}
        return sequence_split, {split: sorted(names) for split, names in assigned.items()}
    weights = {subject: sum(sum(max(0, min(chunk, entries[s]["frames"] - start) - length + 1)
                               for start in range(0, entries[s]["frames"], chunk)) for s in sequences)
               for subject, sequences in subjects.items()}
    items = sorted(subjects)
    tie = np.arange(len(items)); rng.shuffle(tie)
    ranks = {items[int(i)]: rank for rank, i in enumerate(tie)}
    ordered = sorted(items, key=lambda s: (-weights[s], ranks[s]))
    total = sum(weights.values())
    if not total:
        assigned = _assign(items, config["partition"], rng)
    else:
        assigned = {name: [] for name in SPLITS}
        active = [i for i, ratio in enumerate(config["partition"]) if ratio > 0]
        target = np.asarray(config["partition"]) * total
        current = np.zeros(3)
        for index, subject in enumerate(ordered):
            empty = [i for i in active if not assigned[SPLITS[i]]]
            candidates = empty if empty and len(ordered) - index <= len(empty) else active
            i = min(candidates, key=lambda i: ((current[i] + weights[subject]) / max(target[i], np.finfo(float).eps), i))
            assigned[SPLITS[i]].append(subject)
            current[i] += weights[subject]
    sequence_split = {sequence: split for split, names in assigned.items() for subject in names for sequence in subjects[subject]}
    return sequence_split, {split: sorted(names) for split, names in assigned.items()}


def _sequence_arrays(numeric, entry):
    with np.load(_safe(numeric, entry["file"]), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _valid_starts(data, config):
    points, offsets = data["points"], data["offsets"]
    valid = np.asarray([np.isfinite(points[offsets[i]:offsets[i + 1], :3]).all(axis=1).sum()
                        > config["min_valid_points"] for i in range(len(offsets) - 1)])
    length = config["seq_len"]
    timestamps = data.get("timestamps", np.empty(0))
    if length > 1 and not len(timestamps):
        return np.empty(0, dtype=np.int64)
    starts = []
    for start in range(max(0, len(valid) - length + 1)):
        end = start + length
        if not valid[start:end].all():
            continue
        if length > 1:
            gaps = np.abs(np.diff(timestamps[start:end]))
            if len(timestamps[start:end]) != length or not np.isfinite(gaps).all() or not (gaps < config["max_timestamp_gap_ms"]).all():
                continue
        starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def _feature_stats(numeric, entries, aliases, samples, config):
    stats = {"velocity": {"enabled": False, "clip_abs": 1.0},
             "snr": {"enabled": False, "clip_min": 0., "clip_max": 1.}}
    if config["input_dim"] < 4:
        return stats
    refs = sorted({(int(seq), frame) for seq, start, end in samples for frame in range(int(start), int(end))})
    rng = np.random.default_rng(config["seed"])
    velocities, snrs, current, data = [], [], None, None
    for seq, frame_index in refs:
        if current != seq:
            current, data = seq, _sequence_arrays(numeric, entries[aliases[seq]])
        frame = data["points"][data["offsets"][frame_index]:data["offsets"][frame_index + 1]]
        frame = frame[np.isfinite(frame[:, :3]).all(axis=1)]
        if len(frame) > 1024:
            frame = frame[rng.choice(len(frame), size=1024, replace=False)]
        if frame.shape[1] >= 4:
            values = frame[:, 3]; values = values[np.isfinite(values)]
            if len(values): velocities.append(values)
        if config["input_dim"] >= 5 and frame.shape[1] >= 5:
            values = frame[:, 4]; values = values[np.isfinite(values)]
            if len(values): snrs.append(values)
    if velocities:
        stats["velocity"] = {"enabled": True, "clip_abs": max(float(np.quantile(np.abs(np.concatenate(velocities)), .99)), .001)}
    if snrs:
        values = np.concatenate(snrs)
        low, high = float(np.quantile(values, .01)), float(np.quantile(values, .99))
        if high <= low: low, high = float(values.min()), float(values.max() + .001)
        stats["snr"] = {"enabled": True, "clip_min": low, "clip_max": high}
    return stats


def _build_partition(numeric, numeric_manifest, config, descriptor, cache_id, path, progress=None):
    entries = numeric_manifest["sequences"]
    aliases = sorted(entries)
    rng = np.random.default_rng(config["seed"])
    subjects = {split: [] for split in SPLITS}
    assignment = None
    if config["split_strategy"] == "subject":
        assignment, subjects = _weighted_subjects(entries, config, rng)
    samples = {split: [] for split in SPLITS}
    segments = {split: [] for split in SPLITS}
    chunk, length = config["split_chunk_size"], config["seq_len"]
    for seq_index, sequence in enumerate(aliases):
        if progress:
            progress(phase="preparing-cache", progress={"kind": "sequence", "current": seq_index,
                     "total": len(aliases), "label": "Generating split windows"})
        count = int(entries[sequence]["frames"])
        if chunk is None:
            chunks = [(0, count)] if count else []
        else:
            chunks = [(start, min(count, start + chunk)) for start in range(0, count, chunk)]
        selected = _assign(chunks, config["partition"], rng) if assignment is None else {
            split: chunks if assignment[sequence] == split else [] for split in SPLITS}
        valid = _valid_starts(_sequence_arrays(numeric, entries[sequence]), config)
        for split in SPLITS:
            for start, end in selected[split]:
                segments[split].append([sequence, start, end])
                samples[split].extend((seq_index, int(i), int(i + length)) for i in valid[(valid >= start) & (valid + length <= end)])
    for split in SPLITS:
        rng.shuffle(samples[split])
    arrays = {split: np.asarray(values, dtype=np.int32).reshape(-1, 3) for split, values in samples.items()}
    if progress:
        progress(phase="preparing-cache", progress={"kind": "indeterminate", "current": None,
                 "total": None, "label": "Fitting training feature statistics"})
    features = _feature_stats(numeric, entries, aliases, arrays["train"], config)
    _npz(path.with_suffix(".npz"), **arrays)
    metadata = {**_parameters(config), "schema_version": 2, "cache_id": cache_id,
                "algorithm_version": ALGORITHM_VERSION, "sample_dim": config["input_dim"],
                "source_manifest_sha256": descriptor["source_manifest_sha256"],
                "source_inventory_sha256": descriptor["source_inventory_sha256"],
                "numeric_sequences_sha256": _digest({"sequences": entries, "partial": bool(numeric_manifest.get("partial", False))}),
                "sequence_ids": aliases, "split_num_samples": {split: len(a) for split, a in arrays.items()},
                "split_segments": segments, "split_subject_ids": subjects,
                "feature_stats": features, "index_sha256": file_hash(path.with_suffix(".npz")),
                "partial": bool(numeric_manifest.get("partial", False)),
                "validity": "Original chunk/subject assignment, strict valid-point threshold, real timestamp continuity"}
    _json(path, metadata)
    return metadata


def _ready(path, descriptor, cache_id, root=None):
    try:
        metadata = _read(path)
        if not isinstance(metadata, dict): return None
        if (metadata.get("cache_id") != cache_id or metadata.get("source_manifest_sha256") != descriptor["source_manifest_sha256"]
                or metadata.get("source_inventory_sha256") != descriptor["source_inventory_sha256"]
                or metadata.get("algorithm_version") != ALGORITHM_VERSION
                or metadata.get("index_sha256") != file_hash(path.with_suffix(".npz"))):
            return None
        if any(metadata.get(key) != value for key, value in descriptor["parameters"].items()):
            return None
        if root is not None:
            registry = _read(Path(root) / "cache/index.json")
            if not isinstance(registry, dict) or not isinstance(registry.get("entries"), dict):
                return None
            entry = registry["entries"].get(cache_id)
            if not isinstance(entry, dict): return None
            if (not entry or entry.get("metadata_path") != path.relative_to(root).as_posix()
                    or entry.get("metadata_sha256") != file_hash(path)
                    or entry.get("parameters") != descriptor["parameters"]
                    or entry.get("source_manifest_sha256") != descriptor["source_manifest_sha256"]
                    or entry.get("source_inventory_sha256") != descriptor["source_inventory_sha256"]
                    or entry.get("index_sha256") != metadata["index_sha256"]):
                return None
        return metadata
    except (OSError, ValueError, KeyError, TypeError):
        return None


def inspect_partition(data_root, config):
    """Read small manifests only; never prepare or deserialize source PKLs."""
    root = Path(data_root).resolve()
    normalized = normalize_data_config(config)
    manifest, descriptor, cache_id, numeric = _descriptor(root, normalized)
    path = _safe(root, Path("cache/partitions") / cache_id / "splits.json")
    metadata = _ready(path, descriptor, cache_id, root)
    numeric_manifest = numeric / "manifest.json"
    if metadata and numeric != root:
        try:
            numeric_data = _read(numeric_manifest)
            if numeric_data.get("source_manifest_sha256") != descriptor["source_manifest_sha256"]:
                metadata = None
            for item in numeric_data.get("sequences", {}).values():
                array = _safe(numeric, item["file"])
                stat = array.stat()
                if item.get("verified_size") != stat.st_size or item.get("verified_mtime_ns") != stat.st_mtime_ns:
                    metadata = None
                    break
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            metadata = None
    pending = metadata is None
    if pending:
        metadata = {**_parameters(normalized), "seq_len": normalized["seq_len"],
                    "sample_dim": normalized["input_dim"], "split_num_samples": None,
                    "partial": bool(manifest.get("partial", False))}
    return {"numeric_root": numeric, "metadata_path": path, "metadata": metadata,
            "cache_id": cache_id, "pending": pending}


def ensure_partition(data_root, config, progress=None):
    """Resolve the model/strategy/seed combination, preparing only missing caches."""
    root = Path(data_root).resolve()
    normalized = normalize_data_config(config)
    manifest, descriptor, cache_id, numeric = _descriptor(root, normalized)
    if progress:
        progress(phase="preparing-cache", progress={"kind": "indeterminate", "current": None,
                 "total": None, "label": "Checking dataset caches"})
    numeric_manifest = _numeric_sequences(root, manifest, descriptor["source_manifest_sha256"], numeric, progress)
    path = _safe(root, Path("cache/partitions") / cache_id / "splits.json")
    with _lock(path.parent / ".build.lock"):
        metadata = _ready(path, descriptor, cache_id, root)
        numeric_signature = _digest({"sequences": numeric_manifest["sequences"], "partial": bool(numeric_manifest.get("partial", False))})
        if metadata and metadata.get("numeric_sequences_sha256") != numeric_signature:
            metadata = None
        status = "reused" if metadata else "generated"
        if metadata is None:
            print(f"Building {normalized['data_family']} {normalized['split_strategy']} partition (seed {normalized['seed']}).", flush=True)
            metadata = _build_partition(numeric, numeric_manifest, normalized, descriptor, cache_id, path, progress)
    entry = {**descriptor, "kind": "partition", "status": "ready",
             "metadata_path": path.relative_to(root).as_posix(),
             "metadata_sha256": file_hash(path),
             "index_path": path.with_suffix(".npz").relative_to(root).as_posix(),
             "index_sha256": metadata["index_sha256"], "numeric_root": numeric.relative_to(root).as_posix(),
             "split_num_samples": metadata["split_num_samples"]}
    shared = {"root": numeric.relative_to(root).as_posix(), "manifest_sha256": file_hash(numeric / "manifest.json"),
              "source_manifest_sha256": descriptor["source_manifest_sha256"], "sequence_count": len(numeric_manifest["sequences"])}
    _register(root, cache_id, entry, shared)
    if progress:
        progress(phase="loading-data", progress={"kind": "indeterminate", "current": None,
                 "total": None, "label": "Loading prepared dataset"})
    return {"numeric_root": numeric, "metadata_path": path, "metadata": metadata,
            "cache_id": cache_id, "pending": False, "status": status}
