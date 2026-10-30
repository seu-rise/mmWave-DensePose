"""Paper dataset loading without sensor or desktop dependencies."""

__all__ = ["PoseDataset"]


def __getattr__(name):
    if name == "PoseDataset":
        from .dataset import PoseDataset
        return PoseDataset
    raise AttributeError(name)
