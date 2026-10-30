"""Convert dataset skeletons into a model-independent avatar pose protocol."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def convert_kinect_frame_25_to_19(frame):
    """Preserve 19-joint poses or merge the hand joints of a 25-joint pose."""
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"Skeleton must be 2D, got {frame.shape}")
    if frame.shape[0] == 19:
        return frame.copy()
    if frame.shape[0] != 25:
        raise ValueError(f"Expected 19 or 25 joints, got {frame.shape[0]}")
    left, right = (6, 7, 21, 22), (10, 11, 23, 24)
    hands = {6: frame[list(left)].mean(axis=0), 10: frame[list(right)].mean(axis=0)}
    return np.asarray([hands[index] if index in hands else frame[index]
                       for index in range(25)
                       if index not in (*left[1:], *right[1:])], dtype=np.float64)


JOINT_NAMES_19: Tuple[str, ...] = (
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder",
)
JOINT_INDEX_19 = {name: index for index, name in enumerate(JOINT_NAMES_19)}


BONE_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "lower_spine": ("SpineBase", "SpineMid"),
    "upper_spine": ("SpineMid", "SpineShoulder"),
    "neck": ("SpineShoulder", "Neck"),
    "head": ("Neck", "Head"),
    "left_shoulder": ("SpineShoulder", "ShoulderLeft"),
    "left_upper_arm": ("ShoulderLeft", "ElbowLeft"),
    "left_lower_arm": ("ElbowLeft", "HandLeft"),
    "right_shoulder": ("SpineShoulder", "ShoulderRight"),
    "right_upper_arm": ("ShoulderRight", "ElbowRight"),
    "right_lower_arm": ("ElbowRight", "HandRight"),
    "left_upper_leg": ("HipLeft", "KneeLeft"),
    "left_lower_leg": ("KneeLeft", "AnkleLeft"),
    "left_foot": ("AnkleLeft", "FootLeft"),
    "right_upper_leg": ("HipRight", "KneeRight"),
    "right_lower_leg": ("KneeRight", "AnkleRight"),
    "right_foot": ("AnkleRight", "FootRight"),
}


_FOOT_REFERENCE_DIRECTION = np.asarray([0.0, -0.8, -0.6], dtype=np.float64)
_FOOT_REFERENCE_DIRECTION /= np.linalg.norm(_FOOT_REFERENCE_DIRECTION)
REFERENCE_DIRECTIONS: Dict[str, np.ndarray] = {
    "lower_spine": np.asarray([0.0, 0.0, 1.0]),
    "upper_spine": np.asarray([0.0, 0.0, 1.0]),
    "neck": np.asarray([0.0, 0.0, 1.0]),
    "head": np.asarray([0.0, 0.0, 1.0]),
    "left_shoulder": np.asarray([1.0, 0.0, 0.0]),
    "left_upper_arm": np.asarray([1.0, 0.0, 0.0]),
    "left_lower_arm": np.asarray([1.0, 0.0, 0.0]),
    "right_shoulder": np.asarray([-1.0, 0.0, 0.0]),
    "right_upper_arm": np.asarray([-1.0, 0.0, 0.0]),
    "right_lower_arm": np.asarray([-1.0, 0.0, 0.0]),
    "left_upper_leg": np.asarray([0.0, 0.0, -1.0]),
    "left_lower_leg": np.asarray([0.0, 0.0, -1.0]),
    "left_foot": _FOOT_REFERENCE_DIRECTION.copy(),
    "right_upper_leg": np.asarray([0.0, 0.0, -1.0]),
    "right_lower_leg": np.asarray([0.0, 0.0, -1.0]),
    "right_foot": _FOOT_REFERENCE_DIRECTION.copy(),
}


DEFAULT_REFERENCE_LENGTHS: Dict[str, float] = {
    "lower_spine": 0.22,
    "upper_spine": 0.28,
    "neck": 0.12,
    "head": 0.20,
    "left_shoulder": 0.18,
    "left_upper_arm": 0.30,
    "left_lower_arm": 0.27,
    "right_shoulder": 0.18,
    "right_upper_arm": 0.30,
    "right_lower_arm": 0.27,
    "left_upper_leg": 0.45,
    "left_lower_leg": 0.43,
    "left_foot": 0.20,
    "right_upper_leg": 0.45,
    "right_lower_leg": 0.43,
    "right_foot": 0.20,
}


REFERENCE_BODY_HEIGHT_M = (
    DEFAULT_REFERENCE_LENGTHS["lower_spine"]
    + DEFAULT_REFERENCE_LENGTHS["upper_spine"]
    + DEFAULT_REFERENCE_LENGTHS["neck"]
    + DEFAULT_REFERENCE_LENGTHS["head"]
    + 0.5 * (
        DEFAULT_REFERENCE_LENGTHS["left_upper_leg"]
        + DEFAULT_REFERENCE_LENGTHS["right_upper_leg"]
    )
    + 0.5 * (
        DEFAULT_REFERENCE_LENGTHS["left_lower_leg"]
        + DEFAULT_REFERENCE_LENGTHS["right_lower_leg"]
    )
    + abs(float(_FOOT_REFERENCE_DIRECTION[2])) * 0.5 * (
        DEFAULT_REFERENCE_LENGTHS["left_foot"]
        + DEFAULT_REFERENCE_LENGTHS["right_foot"]
    )
)


BEND_CHAINS: Dict[str, Tuple[str, str]] = {
    "left_arm": ("left_upper_arm", "left_lower_arm"),
    "right_arm": ("right_upper_arm", "right_lower_arm"),
    "left_leg": ("left_upper_leg", "left_lower_leg"),
    "right_leg": ("right_upper_leg", "right_lower_leg"),
}


def _unit(vector: np.ndarray, epsilon: float) -> Optional[np.ndarray]:
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    length = float(np.linalg.norm(vector))
    if length <= epsilon:
        return None
    return vector / length


def _quaternion_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-10:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot <= -1.0 + 1e-10:

        candidates = np.eye(3, dtype=np.float64)
        axis_seed = candidates[int(np.argmin(np.abs(candidates @ source)))]
        axis = np.cross(source, axis_seed)
        axis /= np.linalg.norm(axis)
        return np.asarray([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    cross = np.cross(source, target)
    quaternion = np.asarray([cross[0], cross[1], cross[2], 1.0 + dot], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class AvatarPoseService:

    def __init__(
        self,
        reference_bone_lengths: Optional[Mapping[str, float]] = None,
        scale_smoothing: float = 0.15,
        scale_window_size: int = 31,
        min_scale: float = 0.25,
        max_scale: float = 4.0,
        epsilon: float = 1e-6,
        performance_window_size: int = 300,
    ) -> None:
        if not 0.0 < scale_smoothing <= 1.0:
            raise ValueError("scale_smoothing must be in (0, 1]")
        if scale_window_size <= 0 or performance_window_size <= 0:
            raise ValueError("Window sizes must be positive integers")
        if not 0.0 < min_scale <= max_scale:
            raise ValueError("Invalid scale bounds")

        lengths = dict(DEFAULT_REFERENCE_LENGTHS)
        if reference_bone_lengths:
            for bone_name, length in reference_bone_lengths.items():
                if bone_name not in BONE_ENDPOINTS:
                    raise KeyError(f"Unknown reference bone: {bone_name}")
                if not math.isfinite(float(length)) or float(length) <= 0.0:
                    raise ValueError(f"Reference bone length must be positive and finite: {bone_name}")
                lengths[bone_name] = float(length)

        self._reference_lengths = lengths
        self._scale_smoothing = float(scale_smoothing)
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._epsilon = float(epsilon)
        self._scale_candidates: deque[float] = deque(maxlen=int(scale_window_size))
        self._conversion_times_ms: deque[float] = deque(maxlen=int(performance_window_size))
        self._last_directions: Dict[str, np.ndarray] = {}
        self._last_bend_normals: Dict[str, np.ndarray] = {}
        self._scale: Optional[float] = None
        self._invalid_frames = 0
        self._fallback_bone_count = 0

    def reset(self) -> None:
        self._scale_candidates.clear()
        self._conversion_times_ms.clear()
        self._last_directions.clear()
        self._last_bend_normals.clear()
        self._scale = None
        self._invalid_frames = 0
        self._fallback_bone_count = 0

    def convert(
        self,
        keypoints: Sequence[Sequence[float]],
        source: str = "auto",
        frame_id: Any = None,
        timestamp_ns: Any = None,
    ) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()
        inferred_source = str(source or "auto")
        try:
            raw_points = np.asarray(keypoints, dtype=np.float64)
            joint_count = int(raw_points.shape[0]) if raw_points.ndim >= 1 else 0
            if inferred_source == "auto":
                inferred_source = "kinect_gt" if joint_count == 25 else "inference"
            points = convert_kinect_frame_25_to_19(raw_points)
            if points.shape[1] < 3:
                raise ValueError(f"Keypoints require at least three XYZ columns; shape={points.shape}")
            points = np.asarray(points[:, :3], dtype=np.float64)
            if points.shape != (19, 3):
                raise ValueError(f"Keypoints must have shape [19,3] or [25,3+]; shape={raw_points.shape}")
            result = self._convert_valid_shape(points, inferred_source, frame_id, timestamp_ns)
        except (TypeError, ValueError, IndexError) as exc:
            self._invalid_frames += 1
            result = self._invalid_result(inferred_source, frame_id, timestamp_ns, str(exc))

        conversion_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        result["perf"] = {"conversion_ms": float(conversion_ms)}
        self._conversion_times_ms.append(float(conversion_ms))
        return result

    def _convert_valid_shape(
        self,
        points: np.ndarray,
        source: str,
        frame_id: Any,
        timestamp_ns: Any,
    ) -> Dict[str, Any]:
        directions: Dict[str, np.ndarray] = {}
        lengths: Dict[str, float] = {}
        fallback_bones = []

        for bone_name, (start_name, end_name) in BONE_ENDPOINTS.items():
            start_point = points[JOINT_INDEX_19[start_name]]
            end_point = points[JOINT_INDEX_19[end_name]]
            vector = end_point - start_point
            direction = _unit(vector, self._epsilon)
            if direction is None:
                fallback_bones.append(bone_name)
                direction = self._last_directions.get(bone_name, REFERENCE_DIRECTIONS[bone_name]).copy()
            else:
                lengths[bone_name] = float(np.linalg.norm(vector))
                self._last_directions[bone_name] = direction.copy()
            directions[bone_name] = direction

        self._fallback_bone_count += len(fallback_bones)
        scale = self._update_scale(lengths)
        root = points[JOINT_INDEX_19["SpineBase"]]
        root_valid = bool(np.all(np.isfinite(root)))
        if not root_valid:
            root = np.zeros(3, dtype=np.float64)


        valid = root_valid and len(fallback_bones) < (len(BONE_ENDPOINTS) // 2)
        if not valid:
            self._invalid_frames += 1

        foot_points = points[[JOINT_INDEX_19["FootLeft"], JOINT_INDEX_19["FootRight"]]]
        finite_foot_levels = foot_points[np.isfinite(foot_points[:, 2]), 2]

        foot_level_z = float(np.min(finite_foot_levels)) if finite_foot_levels.size else float(root[2])

        bend_normals, fallback_bends = self._calculate_bend_normals(directions)
        quaternions = {
            bone_name: _quaternion_from_to(REFERENCE_DIRECTIONS[bone_name], direction)
            for bone_name, direction in directions.items()
        }
        return {
            "protocol": "avatar_pose/v1",
            "valid": bool(valid),
            "source": source,
            "frame": _json_scalar(frame_id),
            "timestamp": _json_scalar(timestamp_ns),
            "coordinate_frame": "world_z_up",
            "root_position": root.astype(float).tolist(),
            "scale": float(scale),
            "target_height_m": float(REFERENCE_BODY_HEIGHT_M * scale),
            "foot_level_z": foot_level_z,
            "target_directions": {
                name: value.astype(float).tolist() for name, value in directions.items()
            },
            "bend_normals": {
                name: value.astype(float).tolist() for name, value in bend_normals.items()
            },
            "world_delta_quaternions": {
                name: value.astype(float).tolist() for name, value in quaternions.items()
            },
            "fallback_bones": fallback_bones,
            "fallback_bends": fallback_bends,
        }

    def _update_scale(self, lengths: Mapping[str, float]) -> float:
        ratios = [
            length / self._reference_lengths[bone_name]
            for bone_name, length in lengths.items()
            if math.isfinite(length) and length > self._epsilon
        ]
        if ratios:
            frame_scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
            self._scale_candidates.append(float(np.clip(frame_scale, self._min_scale, self._max_scale)))

        robust_scale = (
            float(np.median(np.asarray(self._scale_candidates, dtype=np.float64)))
            if self._scale_candidates else 1.0
        )
        if self._scale is None:
            self._scale = robust_scale
        else:
            alpha = self._scale_smoothing
            self._scale = (1.0 - alpha) * self._scale + alpha * robust_scale
        self._scale = float(np.clip(self._scale, self._min_scale, self._max_scale))
        return self._scale

    def _calculate_bend_normals(
        self,
        directions: Mapping[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], list[str]]:
        normals: Dict[str, np.ndarray] = {}
        fallback_bends = []
        for bend_name, (upper_name, lower_name) in BEND_CHAINS.items():
            normal = _unit(np.cross(directions[upper_name], directions[lower_name]), self._epsilon)
            if normal is None:
                fallback_bends.append(bend_name)
                normal = self._last_bend_normals.get(bend_name)
                if normal is None:

                    normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                self._last_bend_normals[bend_name] = normal.copy()
            normals[bend_name] = normal
        return normals, fallback_bends

    def _invalid_result(
        self,
        source: str,
        frame_id: Any,
        timestamp_ns: Any,
        reason: str,
    ) -> Dict[str, Any]:
        directions = {
            name: self._last_directions.get(name, REFERENCE_DIRECTIONS[name]).astype(float).tolist()
            for name in BONE_ENDPOINTS
        }
        quaternions = {
            name: _quaternion_from_to(REFERENCE_DIRECTIONS[name], np.asarray(direction)).astype(float).tolist()
            for name, direction in directions.items()
        }
        return {
            "protocol": "avatar_pose/v1",
            "valid": False,
            "source": source,
            "frame": _json_scalar(frame_id),
            "timestamp": _json_scalar(timestamp_ns),
            "coordinate_frame": "world_z_up",
            "root_position": [0.0, 0.0, 0.0],
            "scale": float(self._scale if self._scale is not None else 1.0),
            "target_height_m": float(
                REFERENCE_BODY_HEIGHT_M * (self._scale if self._scale is not None else 1.0)
            ),
            "foot_level_z": 0.0,
            "target_directions": directions,
            "bend_normals": {
                name: self._last_bend_normals.get(name, np.asarray([0.0, 1.0, 0.0])).astype(float).tolist()
                for name in BEND_CHAINS
            },
            "world_delta_quaternions": quaternions,
            "fallback_bones": list(BONE_ENDPOINTS),
            "fallback_bends": list(BEND_CHAINS),
            "error": reason,
        }

    def get_performance_snapshot(self) -> Dict[str, Any]:
        samples = np.asarray(self._conversion_times_ms, dtype=np.float64)
        if samples.size == 0:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
                "invalid_frames": int(self._invalid_frames),
                "fallback_bones": int(self._fallback_bone_count),
            }
        return {
            "count": int(samples.size),
            "mean_ms": float(np.mean(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "max_ms": float(np.max(samples)),
            "invalid_frames": int(self._invalid_frames),
            "fallback_bones": int(self._fallback_bone_count),
        }


__all__ = [
    "AvatarPoseService",
    "BEND_CHAINS",
    "BONE_ENDPOINTS",
    "DEFAULT_REFERENCE_LENGTHS",
    "JOINT_NAMES_19",
    "REFERENCE_DIRECTIONS",
]
