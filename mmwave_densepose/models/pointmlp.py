import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn
from .base import KeypointModel
from .pointnet2 import resolve_point_input_dims, split_batched_point_inputs

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class LocalGrouper(nn.Module):

    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize='center', **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ['center', 'anchor']:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        (B, N, C) = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        sample_ratio = S / N
        xbatch = torch.arange(B).repeat_interleave(N).to(xyz.device)
        ybatch = torch.arange(B).repeat_interleave(S).to(xyz.device)
        fps_idx = fps(xyz.reshape((B * N, C)), batch=xbatch, ratio=sample_ratio)
        fps_idx = fps_idx - ybatch * N
        fps_idx = fps_idx.reshape((B, S))
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn(xyz.reshape((B * N, C)), new_xyz.reshape((B * S, C)), self.kneighbors, xbatch, ybatch)[1]
        idx = idx - torch.arange(B).repeat_interleave(S * self.kneighbors).to(xyz.device) * N
        idx = idx.reshape((B, S, self.kneighbors))
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == 'center':
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == 'anchor':
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-05)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return (new_xyz, new_points)

class ConvBNReLU1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(out_channels), self.act)

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes1D(nn.Module):

    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias), nn.BatchNorm1d(int(channel * res_expansion)), self.act)
        if groups > 1:
            self.net2 = nn.Sequential(nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, groups=groups, bias=bias), nn.BatchNorm1d(channel), self.act, nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(channel))
        else:
            self.net2 = nn.Sequential(nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(channel))

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):

    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', use_xyz=True):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        (b, n, s, d) = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        (batch_size, _, _) = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PosExtraction(nn.Module):

    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)

class PointMLP(KeypointModel):

    def __init__(self, info, embed_dim=64, groups=1, res_expansion=1, activation='relu', bias=False, use_xyz=True, normalize='anchor', dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], k_neighbors=[12, 12, 12, 12], reducers=[2, 2, 2, 2], **kwargs):
        super().__init__(info)
        self.stages = len(pre_blocks)
        self._num_classes = info['num_classes']
        self._sample_dim = int(info.get('sample_dim', info.get('feature_dim', 3)))
        (_, self._input_feature_dim, _) = resolve_point_input_dims(sample_dim=self._sample_dim, use_xyz_as_attr_when_sample_dim3=True)
        self.points = info['max_points'] * info['stacks']
        self.embedding = ConvBNReLU1D(self._input_feature_dim, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), 'Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers.'
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)
            last_channel = out_channel
        self.act = get_activation(activation)
        self.classifier = lambda k: nn.Sequential(nn.Linear(last_channel, 512), nn.BatchNorm1d(512), self.act, nn.Dropout(0.5), nn.Linear(512, 256), nn.BatchNorm1d(256), self.act, nn.Dropout(0.5), nn.Linear(256, k))
        if self._num_classes is None:
            num_points = info['num_keypoints']
            self._num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = self.classifier(3)
            self.output = torch.nn.ModuleDict(point_branches)
        else:
            self.output = self.classifier(self._num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, -1].contiguous()
        if x.dim() != 3:
            raise ValueError(f'PointMLP expects input shape [B, N, C>=3], got {tuple(x.shape)}')
        (point_features, xyz) = split_batched_point_inputs(x, sample_dim=self._sample_dim, use_xyz_as_attr_when_sample_dim3=True)
        x = self.embedding(point_features.permute(0, 2, 1))
        for i in range(self.stages):
            (xyz, x) = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        if self._num_classes is None:
            y = []
            for i in range(self._num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x)
        return y
