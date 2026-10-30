"""Streaming MLE, proper-rotation PA-MPJPE, PCK and continuous PCK AUC."""

import torch


def procrustes_align(pred, target):
    """Per-pose similarity alignment without reflection, matching the source implementation."""
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("Expected matching [B,J,3] poses")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred
        target = target.to(pred.dtype)
        pm, tm = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
        pc, tc = pred - pm, target - tm
        u, singular, vh = torch.linalg.svd(pc.transpose(1, 2) @ tc, full_matrices=False)
        correction = torch.ones_like(singular)
        correction[:, -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
        rotation = u @ torch.diag_embed(correction) @ vh
        scale = (singular * correction).sum(1) / pc.square().sum((1, 2)).clamp_min(1e-8)
        return scale[:, None, None] * (pc @ rotation) + tm


class PoseMetrics:
    """Accumulate per-sample sums, not unweighted means of batches. Distances are meters."""

    def __init__(self, joint_indices=None, threshold=0.15):
        self.joint_indices = list(range(19)) if joint_indices is None else list(joint_indices)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold, self.count = float(threshold), 0
        self.sums = torch.zeros(4, len(self.joint_indices), dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, target):
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Pose metric inputs must be finite")
        if pred.shape[1] == 19:
            pred = pred[:, self.joint_indices]
        if target.shape[1] == 19:
            target = target[:, self.joint_indices]
        error = torch.linalg.vector_norm(pred - target, dim=-1)
        aligned_error = torch.linalg.vector_norm(procrustes_align(pred, target) - target, dim=-1)
        values = torch.stack((error, aligned_error, (error <= self.threshold).float(),
                              (1 - error / self.threshold).clamp(0, 1)))
        self.sums += values.double().sum(1).cpu()
        self.count += len(pred)

    def result(self):
        if self.count == 0:
            raise ValueError("No samples were evaluated")
        average = self.sums / self.count
        names = ("mle_m", "pa_mpjpe_m", "pck", "auc")
        result = {name: float(average[i].mean()) for i, name in enumerate(names)}
        result.update(mle_cm=result["mle_m"] * 100, pa_mpjpe_cm=result["pa_mpjpe_m"] * 100,
                      pck_percent=result["pck"] * 100, auc_percent=result["auc"] * 100,
                      samples=self.count, joints=len(self.joint_indices), threshold_m=self.threshold)
        per_joint = [{"joint_index": j, **{n: float(average[i, k]) for i, n in enumerate(names)}}
                     for k, j in enumerate(self.joint_indices)]
        return result, per_joint
