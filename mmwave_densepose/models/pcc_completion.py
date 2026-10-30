from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_cluster import fps as torch_cluster_fps
except ImportError:
    torch_cluster_fps = None
from .base import DenseModel

def furthest_point_sample(xyz, npoint):
    (B, N, C) = xyz.shape
    npoint = max(1, min(int(npoint), int(N)))
    device = xyz.device
    if xyz.is_cuda and torch_cluster_fps is not None:
        flat_xyz = xyz.reshape(B * N, C)
        batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
        ratio = min(1.0, float(npoint) / float(N))
        sampled = torch_cluster_fps(flat_xyz, batch=batch, ratio=ratio, random_start=True)
        local_indices = sampled - batch[sampled] * N
        if local_indices.numel() == B * npoint:
            return local_indices.view(B, npoint)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 10000000000.0, dtype=xyz.dtype, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def gather_points(features, idx):
    (B, C, N) = features.shape
    S = idx.shape[1]
    idx = idx.unsqueeze(1).expand(B, C, S)
    new_features = torch.gather(features, 2, idx)
    return new_features

class CrossTransformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def forward(self, src1, src2):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        (b, c, _) = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)
        return src1

class PCTEncoder(nn.Module):

    def __init__(self, channel=64, out_global_dim=128):
        super(PCTEncoder, self).__init__()
        self.channel = channel
        self.out_global_dim = out_global_dim
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        self.sa1 = CrossTransformer(channel, channel)
        self.sa1_1 = CrossTransformer(channel * 2, channel * 2)
        self.sa2 = CrossTransformer(channel * 2, channel * 2)
        self.sa2_1 = CrossTransformer(channel * 4, channel * 4)
        self.sa3 = CrossTransformer(channel * 4, channel * 4)
        self.sa3_1 = CrossTransformer(channel * 8, channel * 8)
        self.proj_global = nn.Conv1d(channel * 8, out_global_dim, kernel_size=1)
        self.relu = nn.GELU()

    def forward(self, points):
        (batch_size, _, N) = points.size()
        x = self.relu(self.conv1(points))
        x0 = self.conv2(x)
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0)
        x1 = torch.cat([x_g0, x1], dim=1)
        x1 = self.sa1_1(x1, x1)
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1)
        x2 = torch.cat([x_g1, x2], dim=1)
        x2 = self.sa2_1(x2, x2)
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        x3 = self.sa3(x_g2, x2)
        x3 = torch.cat([x_g2, x3], dim=1)
        x3 = self.sa3_1(x3, x3)
        x_g = F.adaptive_max_pool1d(x3, 1)
        x_g = self.proj_global(x_g)
        return x_g

class PCTRefine(nn.Module):

    def __init__(self, channel=128, ratio=1):
        super(PCTRefine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(128, channel, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.sa1 = CrossTransformer(channel * 2, 256)
        self.sa2 = CrossTransformer(256, 256)
        self.sa3 = CrossTransformer(256, channel * ratio)
        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel * 2, channel, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

    def forward(self, coarse, feat_g):
        (batch_size, _, N) = coarse.size()
        y = self.conv_x1(self.relu(self.conv_x(coarse)))
        feat_g = self.conv_1(feat_g)
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)
        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)
        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)
        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)
        return x

class PCCNet(nn.Module):

    def __init__(self):
        super(PCCNet, self).__init__()
        self.encoder = PCTEncoder()
        self.refine = PCTRefine(channel=128, ratio=3)

    def forward(self, x):
        feat_g = self.encoder(x)
        coarse_points = gather_points(x, furthest_point_sample(x.transpose(1, 2).contiguous(), 64))
        completed_points = self.refine(coarse_points, feat_g)
        return completed_points

class PointCloudCompletion(nn.Module):

    def __init__(self, input_points=80, seed_points=24, output_points=196):
        super().__init__()
        self.input_pts = int(input_points)
        self.seed_pts = int(seed_points)
        self.output_pts = int(output_points)
        self.backbone = PCCNet()

    def forward(self, x):
        if x.dim() != 3 or x.shape[-1] < 3:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        x_in = x[:, :, :3].transpose(1, 2).contiguous()
        out = self.backbone(x_in)
        if out.shape[-1] != self.output_pts:
            out = F.interpolate(out, size=self.output_pts, mode='linear', align_corners=False)
        return out.transpose(1, 2).contiguous()

class PCCCompletion(DenseModel):

    @property
    def task_type(self) -> str:
        return 'dense'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_in_points = int(info.get('num_in_points', info.get('max_points', 300)))
        self._num_out_points = int(info.get('num_out_points', 1200))
        self._seed_points = int(info.get('seed_points', 24))
        self._seed_points = max(4, min(self._seed_points, self._num_out_points))
        self.model = PointCloudCompletion(input_points=self._num_in_points, seed_points=self._seed_points, output_points=self._num_out_points)

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError('Invalid model input or configuration: ')
        if x.dim() == 4:
            x = x[:, -1].contiguous()
        if x.dim() != 3:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        if x.shape[-1] < 3:
            raise ValueError(f'Invalid model input or configuration: {x.shape[-1]}')
        if x.shape[-1] > 3:
            x = x[..., :3]
        (bsz, npts, _) = x.shape
        if npts < self._num_in_points:
            pad = x.new_zeros((bsz, self._num_in_points - npts, 3))
            x = torch.cat([x, pad], dim=1)
        elif npts > self._num_in_points:
            x = x[:, :self._num_in_points, :]
        return x.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._adapt_input(x)
        return self.model(x)
