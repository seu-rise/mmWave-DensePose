"""Paper model registry with a shared [batch, time, points, channels] input.

Public model identifiers follow the paper's names in PascalCase. Checkpoint
parameter paths are independent of the public class name.
"""

from importlib import import_module
from typing import Any

import torch

_MODELS = {
    "PASTNet": ("pastnet", "PASTNet"),
    "PointNet2Baseline": ("pointnet2", "PointNet2Baseline"),
    "DGCNN": ("dgcnn", "DGCNN"),
    "PointTransformer": ("point_transformer", "PointTransformer"),
    "PointMLP": ("pointmlp", "PointMLP"),
    "MMPoseFK": ("mmpose_fk", "MMPoseFK"),
    "DualPathCNNBiLSTM": ("dual_path_cnn_bilstm", "DualPathCNNBiLSTM"),
    "PntPoseAT": ("pntpose_at", "PntPoseAT"),
    "PoseGraphNet": ("pose_graph_net", "PoseGraphNet"),
    "GlobalLocalNetwork": ("global_local_network", "GlobalLocalNetwork"),
    "PCCCompletion": ("pcc_completion", "PCCCompletion"),
}


def list_models() -> list[str]:
    """Return the supported public implementation names."""
    return list(_MODELS)


def build_model(name: str, config: dict[str, Any] | None = None) -> torch.nn.Module:
    """Build a paper network without changing its trained parameter layout.

    Common settings are input_dim=5, num_points=300, seq_len=5 and
    num_joints=19. PASTNet accepts ablation='full', 'spatial_only', or
    'baseline'. Model-specific settings are forwarded to the implementation.
    MMPoseFK returns the 19-joint visualization skeleton; metrics must select
    its supported_joints (the paper's 17-joint FK order).
    PCCCompletion returns a completed point cloud, not a pose.
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(list_models())}")
    settings = dict(config or {})
    input_dim = int(settings.get("input_dim", 5))
    num_points = int(settings.get("num_points", 300))
    num_joints = int(settings.get("num_joints", 19))
    seq_len = int(settings.get("seq_len", 5))
    if input_dim not in (3, 4, 5):
        raise ValueError("input_dim must be 3, 4 or 5 (XYZ, Doppler, SNR).")
    if min(num_points, seq_len, num_joints) <= 0:
        raise ValueError("num_points, seq_len and num_joints must be positive.")
    info = {
        **settings,
        "num_classes": None,
        "sample_dim": input_dim,
        "feature_dim": input_dim,
        "num_keypoints": num_joints,
        "max_points": num_points,
        "radar_point_size": num_points,
        "num_in_points": num_points,
        "seq_len": seq_len,
        "stacks": 1,
    }
    module_name, class_name = _MODELS[name]
    if name == "PASTNet":
        ablation = settings.get("ablation", "full")
        if ablation == "baseline":
            module_name, class_name = _MODELS["PointNet2Baseline"]
        elif ablation == "spatial_only":
            module_name, class_name = "ablations", "_SpatialOnlyNet"
        elif ablation != "full":
            raise ValueError("PASTNet ablation must be full, spatial_only or baseline.")
    cls = getattr(import_module(f".{module_name}", __name__), class_name)
    model = cls(info=info)
    if name == "GlobalLocalNetwork" and settings.get("completion_checkpoint"):
        model.load_completion_checkpoint(settings["completion_checkpoint"])
    return model


def __getattr__(name: str):
    if name not in _MODELS:
        raise AttributeError(name)
    module_name, class_name = _MODELS[name]
    return getattr(import_module(f".{module_name}", __name__), class_name)


__all__ = ["build_model", "list_models", *_MODELS]
