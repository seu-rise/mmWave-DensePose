from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import KeypointModel

class ProjectedPlaneCNN(nn.Module):

    def __init__(self, in_channels: int=3, channels: Tuple[int, int]=(16, 32), dropout: float=0.3, batch_norm_momentum: float=0.05):
        super().__init__()
        layers = []
        current_channels = int(in_channels)
        for out_channels in channels:
            layers.extend([nn.Conv2d(current_channels, int(out_channels), kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(int(out_channels), momentum=float(batch_norm_momentum)), nn.ReLU(inplace=True), nn.Dropout2d(p=float(dropout))])
            current_channels = int(out_channels)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return torch.flatten(feat, start_dim=1)

class DualPathCNNBiLSTM(KeypointModel):

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info.get('num_classes')
        if self._num_classes is not None:
            raise ValueError('Invalid model input or configuration: ')
        self._num_points = int(info.get('num_keypoints', 25))
        self._seq_len = int(info.get('t_max', info.get('seq_len', 12)))
        self._grid_size = int(info.get('dual_path_grid_size', 8))
        self._points_per_frame = int(info.get('dual_path_points_per_frame', self._grid_size * self._grid_size))
        self._cnn_dropout = float(info.get('dual_path_cnn_dropout', 0.3))
        self._fusion_dropout = float(info.get('dropout', info.get('dual_path_fusion_dropout', 0.4)))
        self._lstm_hidden = int(info.get('dual_path_lstm_hidden', 128))
        self._lstm_layers = int(info.get('dual_path_lstm_layers', 1))
        self._bn_momentum = float(info.get('dual_path_batch_norm_momentum', 0.05))
        if self._points_per_frame != self._grid_size * self._grid_size:
            raise ValueError('Invalid model input or configuration: ')
        self.xy_cnn = ProjectedPlaneCNN(in_channels=3, channels=(16, 32), dropout=self._cnn_dropout, batch_norm_momentum=self._bn_momentum)
        self.yz_cnn = ProjectedPlaneCNN(in_channels=3, channels=(16, 32), dropout=self._cnn_dropout, batch_norm_momentum=self._bn_momentum)
        cnn_out_dim = self._infer_cnn_out_dim()
        self.xy_temporal = nn.LSTM(input_size=cnn_out_dim, hidden_size=self._lstm_hidden, num_layers=self._lstm_layers, dropout=self._fusion_dropout if self._lstm_layers > 1 else 0.0, bidirectional=True, batch_first=True)
        self.yz_temporal = nn.LSTM(input_size=cnn_out_dim, hidden_size=self._lstm_hidden, num_layers=self._lstm_layers, dropout=self._fusion_dropout if self._lstm_layers > 1 else 0.0, bidirectional=True, batch_first=True)
        temporal_dim = self._lstm_hidden * 2
        fused_dim = self._seq_len * temporal_dim * 2
        self.fusion_head = nn.Sequential(nn.BatchNorm1d(fused_dim, momentum=self._bn_momentum), nn.Dropout(p=self._fusion_dropout), nn.Linear(fused_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, self._num_points * 3))

    def _infer_cnn_out_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self._grid_size, self._grid_size)
            return int(self.xy_cnn(dummy).shape[-1])

    def _align_time_and_mask(self, points: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.dim() == 3:
            points = points.unsqueeze(1)
        elif points.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(points.shape)}')
        (batch_size, seq_len, _, _) = points.shape
        if valid_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=points.device)
        else:
            valid_mask = valid_mask.to(device=points.device, dtype=torch.bool)
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        if seq_len < self._seq_len:
            pad_shape = (batch_size, self._seq_len - seq_len, points.size(2), points.size(3))
            points = torch.cat([points, points.new_zeros(pad_shape)], dim=1)
            mask_pad = torch.zeros((batch_size, self._seq_len - seq_len), dtype=torch.bool, device=points.device)
            valid_mask = torch.cat([valid_mask, mask_pad], dim=1)
        elif seq_len > self._seq_len:
            points = points[:, :self._seq_len]
            valid_mask = valid_mask[:, :self._seq_len]
        return (points, valid_mask)

    def _sort_points_xyz(self, flat_points: torch.Tensor) -> torch.Tensor:
        (batch_frames, num_points, feat_dim) = flat_points.shape
        if num_points < self._points_per_frame:
            pad = flat_points.new_zeros((batch_frames, self._points_per_frame - num_points, feat_dim))
            flat_points = torch.cat([flat_points, pad], dim=1)
            num_points = int(flat_points.size(1))
        sorted_points = flat_points
        xyz = sorted_points[..., :3]
        valid_points = xyz.abs().sum(dim=-1) > 1e-08
        inf_xyz = torch.full_like(xyz, float('inf'))
        sortable_xyz = torch.where(valid_points.unsqueeze(-1), xyz, inf_xyz)
        sorted_points = torch.cat([sortable_xyz, sorted_points[..., 3:]], dim=-1)
        for coord_dim in (2, 1, 0):
            order = torch.argsort(sorted_points[..., coord_dim], dim=1, stable=True)
            gather_index = order.unsqueeze(-1).expand(-1, -1, sorted_points.size(-1))
            sorted_points = torch.gather(sorted_points, dim=1, index=gather_index)
        sorted_points = sorted_points[:, :self._points_per_frame]
        return torch.nan_to_num(sorted_points, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_projected_planes(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (batch_size, seq_len, num_points, feat_dim) = points.shape
        flat = points.reshape(batch_size * seq_len, num_points, feat_dim)
        flat = self._sort_points_xyz(flat)
        velocity = flat[..., 3] if flat.size(-1) >= 4 else flat.new_zeros(flat.shape[:2])
        xy = torch.stack([flat[..., 0], flat[..., 1], velocity], dim=-1)
        yz = torch.stack([flat[..., 1], flat[..., 2], velocity], dim=-1)
        xy = xy.reshape(batch_size * seq_len, self._grid_size, self._grid_size, 3)
        yz = yz.reshape(batch_size * seq_len, self._grid_size, self._grid_size, 3)
        return (xy.permute(0, 3, 1, 2).contiguous(), yz.permute(0, 3, 1, 2).contiguous())

    def _encode_temporal(self, frame_features: torch.Tensor, valid_mask: torch.Tensor, temporal: nn.LSTM) -> torch.Tensor:
        frame_features = frame_features * valid_mask.unsqueeze(-1).to(dtype=frame_features.dtype)
        lengths = valid_mask.long().sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(frame_features, lengths=lengths, batch_first=True, enforce_sorted=False)
        (temporal_out, _) = temporal(packed)
        (temporal_out, _) = pad_packed_sequence(temporal_out, batch_first=True, total_length=self._seq_len)
        return temporal_out * valid_mask.unsqueeze(-1).to(dtype=temporal_out.dtype)

    def _resolve_target_idx(self, valid_mask: torch.Tensor, target_idx: Optional[torch.Tensor]) -> torch.Tensor:
        (batch_size, seq_len) = valid_mask.shape
        if target_idx is None:
            return valid_mask.long().sum(dim=1).clamp(min=1) - 1
        target_idx = target_idx.to(device=valid_mask.device, dtype=torch.long).view(-1)
        if target_idx.numel() != batch_size:
            raise ValueError(f'target_idx batch size mismatch: expected {batch_size}, got {target_idx.numel()}')
        return torch.clamp(target_idx, min=0, max=seq_len - 1)

    def forward(self, data: torch.Tensor, valid_mask: Optional[torch.Tensor]=None, target_idx: Optional[torch.Tensor]=None, return_dict: bool=False):
        (points, valid_mask) = self._align_time_and_mask(data, valid_mask)
        (batch_size, seq_len) = points.shape[:2]
        (xy_plane, yz_plane) = self._build_projected_planes(points)
        xy_feat = self.xy_cnn(xy_plane).reshape(batch_size, seq_len, -1)
        yz_feat = self.yz_cnn(yz_plane).reshape(batch_size, seq_len, -1)
        xy_temporal = self._encode_temporal(xy_feat, valid_mask, self.xy_temporal)
        yz_temporal = self._encode_temporal(yz_feat, valid_mask, self.yz_temporal)
        fused = torch.cat([xy_temporal, yz_temporal], dim=-1).reshape(batch_size, -1)
        pred_target = self.fusion_head(fused).reshape(batch_size, self._num_points, 3)
        target_idx = self._resolve_target_idx(valid_mask, target_idx)
        if return_dict:
            return {'pred_seq': None, 'pred_target': pred_target, 'target_idx': target_idx, 'valid_mask': valid_mask}
        return pred_target
