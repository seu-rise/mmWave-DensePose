"""Read-only sequence inspection without training, Qt, or image-decoder imports.

PKL is executable serialization, not a safe interchange format. Callers must
obtain explicit trust before loading it; a checksum alone is not consent.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import pickle

import numpy as np


IMAGE_KEYS = ("color_frames", "depth_frames", "infrared_frames", "body_index_frames")
# Source: scripts/visualization/draw_body_network_tools.py, BONE_CONNECTIONS_19.
SKELETON_EDGES_19 = (
    (3, 2), (2, 18), (18, 1), (1, 0), (18, 7), (18, 4),
    (0, 14), (0, 10), (7, 8), (8, 9), (4, 5), (5, 6),
    (14, 15), (15, 16), (16, 17), (10, 11), (11, 12), (12, 13),
)


class LoadCancelled(Exception):
    """A cooperative inspection load was cancelled by its caller."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise LoadCancelled("Sequence loading cancelled")


def _report(progress, percent, message):
    if progress is not None:
        progress(int(percent), message)


def _stream(value, name):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a frame sequence")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise ValueError(f"{name} must not be a scalar")
    return value


def _at(stream, index):
    return stream[index] if index < len(stream) else None


def _numeric_frame(value, name, joints=False):
    if value is None:
        return None if joints else np.empty((0, 3), dtype=np.float32)
    array = np.asarray(value)
    if array.size == 0:
        return None if joints else np.empty((0, 3), dtype=np.float32)
    if array.dtype.kind not in "biuf" or array.ndim != 2:
        raise ValueError(f"{name} must be a real numeric [N,C] array")
    if array.shape[1] < 3 or (joints and array.shape[1] != 3):
        raise ValueError(f"{name} must contain {'exactly' if joints else 'at least'} three coordinates")
    return array


def _timestamp(value):
    if value is None:
        return None
    try:
        scalar = float(value)
    except (ValueError, TypeError):
        return None
    return scalar if np.isfinite(scalar) else None


def _image(value):
    """Return encoded bytes unchanged; never decode or re-encode images."""
    if value is None:
        return None, None
    timestamp = None
    if isinstance(value, dict):
        timestamp = _timestamp(value.get("timestamp"))
        value = value.get("image_data")
    if isinstance(value, bytes):
        return value, timestamp
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value), timestamp
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 1:
        return value.tobytes(), timestamp
    if value is None:
        return None, timestamp
    raise ValueError("Image payload must contain encoded bytes, not decoded pixels or objects")


@dataclass
class SequenceRecord:
    """An in-memory sequence or a read-only numeric NPY mapping.

    Frames align by their original index. Missing streams are not interpolated;
    timestamps are exposed in their original milliseconds without alignment.
    """

    path: Path
    frame_count: int
    summary: dict
    _streams: dict = field(repr=False)
    _offsets: object = field(default=None, repr=False)

    def frame(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.frame_count:
            raise IndexError(f"Frame index must be between 0 and {self.frame_count - 1}")
        if self._offsets is not None:
            begin, end = self._offsets[index:index + 2]
            radar = self._streams["points"][int(begin):int(end)]
        else:
            radar = _numeric_frame(_at(self._streams.get("radar_frames", ()), index), "radar frame")
        joints = _numeric_frame(_at(self._streams.get("keypoint_frames", ()), index),
                                "keypoint frame", joints=True)
        images, image_timestamps = {}, {}
        for key in IMAGE_KEYS:
            images[key], image_timestamps[key] = _image(_at(self._streams.get(key, ()), index))
        return {"radar": radar, "joints": joints,
                "radar_timestamp": _timestamp(_at(self._streams.get("radar_timestamp", ()), index)),
                "kinect_timestamp": _timestamp(_at(self._streams.get("kinect_timestamp", ()), index)),
                "images": images, "image_timestamps": image_timestamps}


def _validate_pickle(path, value, progress, cancelled):
    if not isinstance(value, dict) or not any(key in value for key in
                                             ("radar_frames", "keypoint_frames", *IMAGE_KEYS)):
        raise ValueError("PKL must be a sequence dictionary, not an index or arbitrary object")
    keys = ("radar_frames", "keypoint_frames", "radar_timestamp", "kinect_timestamp", *IMAGE_KEYS)
    streams = {key: _stream(value.get(key), key) for key in keys}
    frame_count = max((len(streams[key]) for key in keys), default=0)
    if frame_count == 0:
        raise ValueError("Sequence contains no frames")
    for index in range(frame_count):
        _check_cancel(cancelled)
        _numeric_frame(_at(streams["radar_frames"], index), "radar frame")
        _numeric_frame(_at(streams["keypoint_frames"], index), "keypoint frame", joints=True)
        for key in IMAGE_KEYS:
            _image(_at(streams[key], index))
        if index % 64 == 0:
            _report(progress, 75 + 24 * index // frame_count, "Validating sequence streams")
    modalities = [key for key in keys if len(streams[key])]
    return SequenceRecord(path, frame_count, {
        "filename": path.name, "frame_count": frame_count, "modalities": modalities,
        "source_kind": "full_multimodal_pkl", "stream_lengths": {
            key: len(streams[key]) for key in keys},
    }, streams)


def _load_npz(path, cancelled):
    with np.load(path, allow_pickle=False) as archive:
        if not {"points", "offsets", "joints"}.issubset(archive.files):
            raise ValueError("NPZ must contain points, offsets, and joints")
        points, offsets, joints = (archive[key] for key in ("points", "offsets", "joints"))
    _check_cancel(cancelled)
    if points.dtype.kind not in "biuf" or points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("NPZ points must be a real numeric [P,C>=3] array")
    if offsets.dtype.kind not in "iu" or offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("NPZ offsets must be a one-dimensional integer [F+1] array")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(points):
        raise ValueError("NPZ offsets must start at zero, increase monotonically, and end at P")
    if np.any(offsets > len(points)):
        raise ValueError("NPZ offsets exceed point bounds")
    count = len(offsets) - 1
    if (joints.dtype.kind not in "biuf" or joints.ndim != 3 or joints.shape[0] != count
            or joints.shape[2] != 3):
        raise ValueError("NPZ joints must be a real numeric [F,J,3] array matching offsets")
    return SequenceRecord(path, count, {"filename": path.name, "frame_count": count,
        "modalities": ["radar_frames", "keypoint_frames"], "source_kind": "numeric_npz_cache"},
        {"points": points, "keypoint_frames": joints}, offsets)


def load_sequence(path, trusted=False, expected_sha256=None, progress=None, cancelled=None):
    """Load a supported local file read-only, with explicit PKL trust and cancellation.

    SHA256 verifies file integrity, not publisher authenticity. Pickle execution
    itself cannot be interrupted safely; cancellation is checked before and after.
    """
    path = Path(path).expanduser().resolve()
    _check_cancel(cancelled)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in (".pkl", ".pickle", ".npz", ".npy"):
        raise ValueError("Supported files: trusted PKL, numeric NPZ, and numeric NPY")
    if suffix in (".pkl", ".pickle") and not trusted:
        raise ValueError("PKL can execute code. Explicit trust is required before loading.")
    _report(progress, 0, "Checking sequence file")
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError("Expected SHA256 must contain 64 hexadecimal characters")
        digest, consumed, size = hashlib.sha256(), 0, path.stat().st_size
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                _check_cancel(cancelled)
                digest.update(chunk)
                consumed += len(chunk)
                _report(progress, 45 * consumed // max(size, 1), "Verifying SHA256")
        _check_cancel(cancelled)
        if digest.hexdigest() != expected:
            raise ValueError("Sequence SHA256 mismatch; file was not loaded")
    _check_cancel(cancelled)
    _report(progress, 50, "Loading sequence (large PKL files may take a while)")
    if suffix in (".pkl", ".pickle"):
        with path.open("rb") as stream:
            value = pickle.load(stream)  # Explicit caller trust was checked above.
        _check_cancel(cancelled)
        record = _validate_pickle(path, value, progress, cancelled)
    elif suffix == ".npz":
        record = _load_npz(path, cancelled)
    else:
        points = np.load(path, allow_pickle=False, mmap_mode="r")
        if points.dtype.kind not in "biuf" or points.ndim != 3 or points.shape[2] < 3 or not len(points):
            raise ValueError("NPY must be a non-empty real numeric radar [F,N,C>=3] array")
        record = SequenceRecord(path, len(points), {"filename": path.name, "frame_count": len(points),
            "modalities": ["radar_frames"], "source_kind": "numeric_npy_radar"},
            {"radar_frames": points})
    _check_cancel(cancelled)
    _report(progress, 100, "Sequence ready")
    return record
