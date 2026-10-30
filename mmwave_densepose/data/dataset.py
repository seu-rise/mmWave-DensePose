"""Load canonical radar/skeleton sequences and frozen paper split indexes."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .cache import training_root, record_config


def normalize_points(frame, input_dim, feature_stats):
    """Apply the original training-only Doppler/SNR clipping statistics."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2 or frame.shape[1] < 3:
        return np.zeros((0, input_dim), dtype=np.float32)
    frame = frame[np.all(np.isfinite(frame[:, :3]), axis=1)]
    points = np.zeros((len(frame), input_dim), dtype=np.float32)
    width = min(frame.shape[1], input_dim)
    points[:, :width] = frame[:, :width]
    np.nan_to_num(points, copy=False, nan=0, posinf=0, neginf=0)
    velocity = feature_stats.get("velocity", {})
    snr = feature_stats.get("snr", {})
    if input_dim >= 4 and velocity.get("enabled", False):
        clip = max(float(velocity["clip_abs"]), 1e-6)
        points[:, 3] = np.clip(points[:, 3], -clip, clip) / clip
    if input_dim >= 5 and snr.get("enabled", False):
        low, high = float(snr["clip_min"]), float(snr["clip_max"])
        points[:, 4] = (np.clip(points[:, 4], low, high) - low) / (high - low) if high > low else 0
    return points


def align_points(points, num_points):
    """Original zero padding and random-start, squared-distance greedy FPS.

    The NumPy RNG is intentionally retained: callers seed their experiment and
    DataLoader workers rather than replacing the trained preprocessing policy.
    """
    count = len(points)
    if count <= num_points:
        result = np.zeros((num_points, points.shape[1]), dtype=np.float32)
        result[:count] = points
        return result
    xyz = points[:, :3]
    indices = np.empty(num_points, dtype=np.int64)
    indices[0] = np.random.randint(count)
    delta = xyz - xyz[indices[0]]
    nearest = np.einsum("ij,ij->i", delta, delta, dtype=np.float32)
    for i in range(1, num_points):
        indices[i] = np.argmax(nearest)
        delta = xyz - xyz[indices[i]]
        distance = np.einsum("ij,ij->i", delta, delta, dtype=np.float32)
        np.minimum(nearest, distance, out=nearest)
    return points[indices]


class PoseDataset(Dataset):
    """Return ``points[T,N,C]``, ``target[19,3]`` and stable ``sample_id``.

    ``split_manifest`` accepts a canonical split name or its JSON path. The
    frozen index determines valid windows; timestamps are never invented.
    """

    def __init__(self, data_root, split_manifest="chunk_s42", split="train",
                 num_points=300, input_dim=5, seq_len=5, cache_size=2, seed=42, prepared=None):
        self.source_root = Path(data_root)
        self.data_root = Path(prepared["numeric_root"]) if prepared else training_root(self.source_root, split_manifest)
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if num_points < 1 or input_dim not in (3, 4, 5) or seq_len < 1 or cache_size < 1:
            raise ValueError("Require positive point/sequence/cache sizes and input_dim 3, 4 or 5")
        path = Path(prepared["metadata_path"]) if prepared else Path(split_manifest)
        if not path.is_file():
            path = self.data_root / "splits" / (str(split_manifest).removesuffix(".json") + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"Missing split {path}; verify the installed dataset and experiment configuration")
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        self.feature_stats = self.metadata["feature_stats"]
        self.cache_identity = prepared["cache_id"] if prepared else record_config(self.source_root, self.data_root,
            {"split_manifest": str(split_manifest), "num_points": num_points, "input_dim": input_dim,
             "seq_len": seq_len, "seed": seed, "num_joints": 19}, self.metadata)
        expected = int(self.metadata["seq_len"])
        if seq_len != expected:
            raise ValueError(f"Split {path.stem} contains {expected}-frame windows, not {seq_len}")
        self.seq_len, self.num_points, self.input_dim = seq_len, num_points, input_dim
        self.cache_size, self.split = cache_size, split
        manifest = json.loads((self.data_root / "manifest.json").read_text(encoding="utf-8"))
        self.sequences = manifest["sequences"]
        self.sequence_ids = self.metadata["sequence_ids"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            self.samples = arrays[split].copy()
        if len(self.samples) != self.metadata["split_num_samples"][split]:
            raise ValueError("Split count does not match metadata")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def _sequence(self, sequence_id):
        if sequence_id not in self._cache:
            entry = self.sequences[sequence_id]
            with np.load(self.data_root / entry["file"], allow_pickle=False) as data:
                self._cache[sequence_id] = tuple(data[k].copy() for k in ("points", "offsets", "joints"))
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(sequence_id)
        return self._cache[sequence_id]

    def __getitem__(self, index):
        sequence_index, start, end = map(int, self.samples[index])
        sequence_id = self.sequence_ids[sequence_index]
        points, offsets, joints = self._sequence(sequence_id)
        frames = [align_points(normalize_points(points[offsets[i]:offsets[i + 1]],
                  self.input_dim, self.feature_stats), self.num_points) for i in range(start, end)]
        return {"points": torch.from_numpy(np.stack(frames)),
                "target": torch.from_numpy(joints[end - 1].copy()),
                "sample_id": f"{sequence_id}:{start}:{end}"}
