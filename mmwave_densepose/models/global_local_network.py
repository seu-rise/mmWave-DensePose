from typing import Tuple
import torch
import torch.nn as nn
from .base import KeypointModel
from .pcc_completion import PointCloudCompletion
DEFAULT_VOXEL_BOUNDS_MIN: Tuple[float, float, float] = (-1.5, 0.0, -0.2)
DEFAULT_VOXEL_BOUNDS_MAX: Tuple[float, float, float] = (1.5, 4.0, 2.2)

class PoseBranch(nn.Module):

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.2), nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2), nn.Conv2d(64, hidden_dim, 3, 1, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x_seq):
        (batch_size, seq_len, channels, height, width) = x_seq.shape
        frame_batch = x_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_features = self.pool(self.cnn(frame_batch)).flatten(1)
        feat_seq = frame_features.view(batch_size, seq_len, -1)
        (lstm_out, _) = self.lstm(feat_seq)
        return lstm_out[:, -1]

def pointcloud_to_grid(pc, grid_size, bounds_min: torch.Tensor, bounds_max: torch.Tensor, padding_eps: float=1e-08):
    (B, N, C) = pc.shape
    device = pc.device
    dtype = pc.dtype
    bounds_min = bounds_min.to(device=device, dtype=dtype).view(1, 1, 3)
    bounds_max = bounds_max.to(device=device, dtype=dtype).view(1, 1, 3)
    span = (bounds_max - bounds_min).clamp_min(1e-06)
    xyz = pc[..., :3]
    normalized_xyz = ((xyz - bounds_min) / span).clamp(0.0, 1.0)
    idx = (normalized_xyz[..., :2] * (grid_size - 1)).long().clamp(0, grid_size - 1)
    pc_features = pc.clone()
    pc_features[..., :3] = normalized_xyz
    finite_mask = torch.isfinite(xyz).all(dim=-1)
    non_padding_mask = torch.any(torch.abs(pc) > float(padding_eps), dim=-1)
    valid_point_mask = finite_mask & non_padding_mask
    if hasattr(torch.Tensor, 'scatter_reduce_'):
        linear_idx = idx[..., 1] * grid_size + idx[..., 0]
        point_order = torch.arange(N, device=device, dtype=torch.long).view(1, N).expand(B, N)
        masked_order = torch.where(valid_point_mask, point_order, torch.full_like(point_order, -1))
        last_point_idx = torch.full((B, grid_size * grid_size), -1, device=device, dtype=torch.long)
        last_point_idx.scatter_reduce_(dim=1, index=linear_idx, src=masked_order, reduce='amax', include_self=True)
        valid_mask = last_point_idx >= 0
        gather_idx = last_point_idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, C)
        selected_points = torch.gather(pc_features, dim=1, index=gather_idx)
        grid_flat = torch.zeros(B, grid_size * grid_size, C, device=device, dtype=pc_features.dtype)
        grid_flat[valid_mask] = selected_points[valid_mask]
        return grid_flat.view(B, grid_size, grid_size, C).permute(0, 3, 1, 2).contiguous()
    grid = torch.zeros(B, C, grid_size, grid_size, device=device, dtype=pc_features.dtype)
    for b in range(B):
        for i in range(N):
            if not bool(valid_point_mask[b, i]):
                continue
            (x, y) = idx[b, i]
            grid[b, :, y, x] = pc_features[b, i, :C]
    return grid.contiguous()

class GlobalLocalPoseEstimation(nn.Module):

    def __init__(self, num_joints=19, seq_len=5, hidden_dim=128, input_points=300, completed_points=196, seed_points=24, use_precomputed_completion=False, bounds_min: Tuple[float, float, float]=DEFAULT_VOXEL_BOUNDS_MIN, bounds_max: Tuple[float, float, float]=DEFAULT_VOXEL_BOUNDS_MAX, padding_eps: float=1e-08):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_joints = int(num_joints)
        self.padding_eps = float(padding_eps)
        self.use_precomputed_completion = bool(use_precomputed_completion)
        bounds_min_tensor = torch.tensor(bounds_min, dtype=torch.float32)
        bounds_max_tensor = torch.tensor(bounds_max, dtype=torch.float32)
        if bounds_min_tensor.shape != (3,) or bounds_max_tensor.shape != (3,):
            raise ValueError('Invalid model input or configuration: ')
        if torch.any(bounds_max_tensor <= bounds_min_tensor):
            raise ValueError('Invalid model input or configuration: ')
        self.register_buffer('bounds_min', bounds_min_tensor)
        self.register_buffer('bounds_max', bounds_max_tensor)
        self.pcc = None
        if not self.use_precomputed_completion:
            self.pcc = PointCloudCompletion(input_points=int(input_points), seed_points=int(seed_points), output_points=int(completed_points))
        self.global_branch = PoseBranch(in_channels=3, hidden_dim=hidden_dim)
        self.local_branch = PoseBranch(in_channels=5, hidden_dim=hidden_dim)
        self.regressor = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, self.num_joints * 3))

    def forward_with_completed(self, raw_radar_seq, completed_seq=None):
        (B, T, N, _) = raw_radar_seq.shape
        if completed_seq is None:
            if self.pcc is None:
                raise ValueError('Invalid model input or configuration: ')
            xyz = raw_radar_seq[:, :, :, :3].reshape(B * T, N, 3).contiguous()
            comp_flat = self.pcc(xyz)
            comp_seq = comp_flat.view(B, T, comp_flat.shape[1], 3).contiguous()
        else:
            if completed_seq.dim() != 4 or completed_seq.shape[:2] != (B, T):
                raise ValueError(f'Invalid model input or configuration: {tuple(raw_radar_seq.shape)}, completed={tuple(completed_seq.shape)}')
            if completed_seq.shape[-1] != 3:
                raise ValueError('Invalid model input or configuration: ')
            comp_seq = completed_seq.contiguous()
            comp_flat = comp_seq.reshape(B * T, comp_seq.shape[2], 3)
        g_grid_flat = pointcloud_to_grid(comp_flat, 14, self.bounds_min, self.bounds_max, padding_eps=self.padding_eps)
        local_flat = raw_radar_seq.reshape(B * T, N, raw_radar_seq.shape[-1]).contiguous()
        l_grid_flat = pointcloud_to_grid(local_flat, 8, self.bounds_min, self.bounds_max, padding_eps=self.padding_eps)
        g_grid = g_grid_flat.view(B, T, *g_grid_flat.shape[1:])
        l_grid = l_grid_flat.view(B, T, *l_grid_flat.shape[1:])
        g_feat = self.global_branch(g_grid)
        l_feat = self.local_branch(l_grid)
        fuse = torch.cat([g_feat, l_feat], dim=-1)
        pose = self.regressor(fuse).view(raw_radar_seq.shape[0], self.num_joints, 3)
        return (pose, comp_seq)

    def forward(self, raw_radar_seq, completed_seq=None):
        (pose, _) = self.forward_with_completed(raw_radar_seq, completed_seq=completed_seq)
        return pose

class GlobalLocalNetwork(KeypointModel):
    """Global/local pose branches with a frozen point completion network.

    Training requires a separately trained completion checkpoint. A complete
    pose checkpoint already contains that branch and can be loaded strictly.
    """

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info.get('num_classes', None)
        if self._num_classes is not None:
            raise ValueError('Invalid model input or configuration: ')
        self._num_points = int(info.get('num_keypoints', 19))
        self._seq_len = int(info.get('seq_len', 5))
        self._n_points_input = int(info.get('max_points', info.get('num_in_points', 300)))
        self._n_points_completed = int(info.get('pcc_output_points', info.get('n_points_completed', info.get('num_out_points', 196))) or 196)
        self._hidden_dim = int(info.get('hidden_dim', 128))
        self._seed_points = int(info.get('seed_points', 24))
        self._use_precomputed_completion = bool(info.get('pcc_cache_file'))
        bounds_min = tuple(info.get('voxel_bounds_min', DEFAULT_VOXEL_BOUNDS_MIN))
        bounds_max = tuple(info.get('voxel_bounds_max', DEFAULT_VOXEL_BOUNDS_MAX))
        self._using_default_voxel_bounds = 'voxel_bounds_min' not in info and 'voxel_bounds_max' not in info
        self.model = GlobalLocalPoseEstimation(num_joints=self._num_points, seq_len=self._seq_len, hidden_dim=self._hidden_dim, input_points=self._n_points_input, completed_points=self._n_points_completed, seed_points=self._seed_points, use_precomputed_completion=self._use_precomputed_completion, bounds_min=(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])), bounds_max=(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])))
        if self.model.pcc is not None:
            self._freeze_pcc_parameters()

    def load_completion_checkpoint(self, path: str) -> None:
        """Load all completion parameters from a prepared weights-only file."""
        if self.model.pcc is None:
            raise ValueError("A precomputed completion cache disables the PCC branch.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        name = checkpoint.get("model_name", checkpoint.get("model"))
        if name and name != "PCCCompletion":
            raise ValueError(f"Completion checkpoint must be PCCCompletion, got {name!r}")
        state = checkpoint.get("state_dict", checkpoint)
        target = self.model.pcc.state_dict()
        for prefix in ("", "model.", "model.model."):
            candidate = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if set(candidate) == set(target):
                self.model.pcc.load_state_dict(candidate, strict=True)
                self._freeze_pcc_parameters()
                return
        raise ValueError("Completion checkpoint does not contain the exact PCC parameter set.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model.pcc is not None:
            self.model.pcc.eval()
        return self

    def _freeze_pcc_parameters(self) -> None:
        if self.model.pcc is None:
            return
        for parameter in self.model.pcc.parameters():
            parameter.requires_grad = False
        self.model.pcc.eval()

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError('Invalid model input or configuration: ')
        if x.dim() == 3:
            x = x.unsqueeze(1).repeat(1, self._seq_len, 1, 1)
        elif x.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        (B, T, N, C) = x.shape
        if T < self._seq_len:
            pad_t = x[:, -1:, :, :].repeat(1, self._seq_len - T, 1, 1)
            x = torch.cat([x, pad_t], dim=1)
        elif T > self._seq_len:
            x = x[:, :self._seq_len, :, :]
        if N < self._n_points_input:
            pad_n = x.new_zeros((B, x.shape[1], self._n_points_input - N, C))
            x = torch.cat([x, pad_n], dim=2)
        elif N > self._n_points_input:
            x = x[:, :, :self._n_points_input, :]
        C = x.shape[-1]
        if C < 3:
            raise ValueError('Invalid model input or configuration: ')
        if C < 5:
            pad_c = x.new_zeros((*x.shape[:-1], 5 - C))
            x = torch.cat([x, pad_c], dim=-1)
        elif C > 5:
            x = x[..., :5]
        return x.contiguous()

    def _adapt_completed_input(self, completed_points: torch.Tensor) -> torch.Tensor:
        if not isinstance(completed_points, torch.Tensor):
            raise ValueError('Invalid model input or configuration: ')
        if completed_points.dim() == 3:
            completed_points = completed_points.unsqueeze(1).repeat(1, self._seq_len, 1, 1)
        elif completed_points.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(completed_points.shape)}')
        time_length = int(completed_points.shape[1])
        if time_length < self._seq_len:
            pad = completed_points[:, -1:, :, :].repeat(1, self._seq_len - time_length, 1, 1)
            completed_points = torch.cat([completed_points, pad], dim=1)
        elif time_length > self._seq_len:
            completed_points = completed_points[:, :self._seq_len, :, :]
        if completed_points.shape[-1] != 3:
            raise ValueError('Invalid model input or configuration: ')
        return completed_points.contiguous()

    def forward_with_completed(self, x, completed_points=None):
        x = self._adapt_input(x)
        if completed_points is not None:
            completed_points = self._adapt_completed_input(completed_points)
        (y, completed_seq) = self.model.forward_with_completed(x, completed_seq=completed_points)
        if y.dim() == 2:
            y = y.view(y.shape[0], self._num_points, 3)
        return (y, completed_seq)

    def forward(self, x, completed_points=None):
        (y, _) = self.forward_with_completed(x, completed_points=completed_points)
        return y
