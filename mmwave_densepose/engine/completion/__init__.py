"""PCC-only completion supervision and Euclidean Chamfer distance."""

import torch


def chamfer_per_cloud(prediction, target):
    """Return the source PCC bidirectional, unsquared Chamfer distance in metres.

    Prediction and target may have different point counts (196 and 384 in the
    paper configuration). Each direction is averaged separately, then summed.
    """
    if (prediction.ndim != 3 or target.ndim != 3
            or prediction.shape[0] != target.shape[0]
            or prediction.shape[-1] != 3 or target.shape[-1] != 3
            or min(prediction.shape[:2]) < 1 or target.shape[1] < 1):
        raise ValueError("Chamfer distance requires nonempty [B,N,3] and [B,M,3] clouds")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Chamfer distance requires finite coordinates")
    distances = torch.cdist(prediction, target, p=2)
    return distances.min(dim=2).values.mean(dim=1) + distances.min(dim=1).values.mean(dim=1)


def chamfer_loss(prediction, target):
    return chamfer_per_cloud(prediction, target).mean()
