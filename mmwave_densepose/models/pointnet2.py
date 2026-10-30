from typing import Optional, Tuple
import torch
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from .base import KeypointModel

def _safe_global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f'_safe_global_max_pool expects x as [N, C], got {tuple(x.shape)}')
    batch = batch.reshape(-1).to(device=x.device, dtype=torch.long).contiguous()
    if batch.numel() != x.size(0):
        raise ValueError(f'_safe_global_max_pool expects batch length == N, got {batch.numel()} vs {x.size(0)}')
    if x.size(0) <= 0:
        return x.new_zeros((0, x.size(-1)))
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return global_max_pool(x, batch)
    except RuntimeError as err:
        if 'illegal memory access' not in str(err).lower():
            raise
    valid_mask = batch >= 0
    if not bool(valid_mask.any()):
        return x.new_zeros((0, x.size(-1)))
    x = x[valid_mask]
    batch = batch[valid_mask]
    out = x.new_full((int(batch.max().item()) + 1, x.size(-1)), float('-inf'))
    index = batch.view(-1, 1).expand(-1, x.size(-1))
    out.scatter_reduce_(0, index, x, reduce='amax', include_self=True)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def resolve_point_input_dims(sample_dim: int=3, use_xyz_as_attr_when_sample_dim3: bool=False) -> Tuple[int, int, bool]:
    sample_dim = max(3, int(sample_dim))
    attr_dim = max(0, sample_dim - 3)
    use_xyz_as_attr = bool(use_xyz_as_attr_when_sample_dim3) and attr_dim <= 0
    if attr_dim > 0:
        effective_attr_dim = attr_dim
    elif use_xyz_as_attr:
        effective_attr_dim = 3
    else:
        effective_attr_dim = 0
    return (sample_dim, effective_attr_dim, use_xyz_as_attr)

def split_batched_point_inputs(data: torch.Tensor, sample_dim: int=3, use_xyz_as_attr_when_sample_dim3: bool=False) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if data.dim() != 3 or data.size(-1) < 3:
        raise ValueError(f'point input expects shape [B, N, C>=3], got {tuple(data.shape)}')
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    (batch_size, num_points, feat_dim) = data.shape
    (sample_dim, effective_attr_dim, use_xyz_as_attr) = resolve_point_input_dims(sample_dim=sample_dim, use_xyz_as_attr_when_sample_dim3=use_xyz_as_attr_when_sample_dim3)
    pos = data[:, :, :3]
    if effective_attr_dim <= 0:
        return (None, pos)
    if use_xyz_as_attr:
        attr = data[:, :, :3]
    else:
        attr_end = min(feat_dim, sample_dim)
        attr = data[:, :, 3:attr_end]
        if attr.size(-1) < effective_attr_dim:
            pad = attr.new_zeros(batch_size, num_points, effective_attr_dim - attr.size(-1))
            attr = torch.cat([attr, pad], dim=-1)
    return (attr.contiguous(), pos.contiguous())

class SAModule(torch.nn.Module):

    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        batch = batch.to(device=pos.device, dtype=torch.long).contiguous()
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        if x is not None:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        if pos.size(0) <= 0 or batch.numel() <= 0:
            return (x, pos, batch)
        idx = fps(pos, batch, ratio=self.ratio)
        (row, col) = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        (pos, batch) = (pos[idx], batch[idx])
        return (x, pos, batch)

class GlobalSAModule(torch.nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        if x is None:
            x = self.nn(pos)
        else:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            x = self.nn(torch.cat([x, pos], dim=1))
        x = _safe_global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return (x, pos, batch)

class PointNetBackbone(torch.nn.Module):

    def __init__(self, sample_dim: int=3, use_xyz_as_attr_when_sample_dim3: bool=False):
        super().__init__()
        (self.sample_dim, self._effective_attr_dim, self._use_xyz_as_attr) = resolve_point_input_dims(sample_dim=sample_dim, use_xyz_as_attr_when_sample_dim3=use_xyz_as_attr_when_sample_dim3)
        sa1_in_dim = 3 + self._effective_attr_dim if self._effective_attr_dim > 0 else 3
        self.sa1_module = SAModule(0.5, 0.2, MLP([sa1_in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = data.shape[0]
        num_points = data.shape[1]
        (x, pos) = split_batched_point_inputs(data, sample_dim=self.sample_dim, use_xyz_as_attr_when_sample_dim3=self._use_xyz_as_attr)
        pos = pos.reshape(batch_size * num_points, 3)
        if x is not None:
            x = x.reshape(batch_size * num_points, self._effective_attr_dim)
        batch = torch.arange(batch_size, device=data.device).repeat_interleave(num_points)
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        (x, _, _) = sa3_out
        return x

class PointNet2Baseline(KeypointModel):

    def __init__(self, info=None):
        super().__init__(info)
        self._num_classes = info['num_classes']
        self._sample_dim = int(info.get('sample_dim', info.get('feature_dim', 3)))
        self.backbone = PointNetBackbone(sample_dim=self._sample_dim, use_xyz_as_attr_when_sample_dim3=True)
        if self._num_classes is None:
            self._num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self._num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self._num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        if data.dim() == 4:
            data = data[:, -1].contiguous()
        if data.dim() != 3:
            raise ValueError(f'PointNet2Baseline expects input shape [B, N, C>=3], got {tuple(data.shape)}')
        x = self.backbone(data)
        if self._num_classes is None:
            y = []
            for i in range(self._num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y
