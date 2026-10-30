from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import KeypointModel
JOINT_NAMES_19: List[str] = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder']
JOINT_NAMES_17: List[str] = ['SpineBase', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'SpineMid', 'SpineShoulder', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'ShoulderRight', 'ElbowRight', 'WristRight']
POSE17_FROM_19: List[int] = [0, 10, 11, 12, 14, 15, 16, 1, 18, 2, 3, 4, 5, 6, 7, 8, 9]
POSE19_FROM_17: Dict[int, int] = {0: 0, 1: 7, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 1, 11: 2, 12: 3, 14: 4, 15: 5, 16: 6, 18: 8}
PARENT_INDICES_17: List[int] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
FK_CHILD_ORDER_17: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ROTATABLE_JOINTS_17: List[int] = [2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
BASE_DIRS_17: List[List[float]] = [[0.0, 0.0, 0.0], [0.9139, -0.3875, 0.1207], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [-0.9248, -0.3556, 0.1354], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
FOOT_OFFSETS_FROM_ANKLE: List[List[float]] = [[-0.0069, -0.0885, -0.0622], [-0.0061, -0.0963, -0.056]]
DEFAULT_VOXEL_BOUNDS_MIN: Tuple[float, float, float] = (-1.5, 0.0, -0.2)
DEFAULT_VOXEL_BOUNDS_MAX: Tuple[float, float, float] = (1.5, 4.0, 2.2)

class RadarPointCloudVoxelization(nn.Module):

    def __init__(self, grid_shape: Tuple[int, int, int]=(32, 32, 64), bounds_min: Tuple[float, float, float]=DEFAULT_VOXEL_BOUNDS_MIN, bounds_max: Tuple[float, float, float]=DEFAULT_VOXEL_BOUNDS_MAX, eps: float=1e-06, padding_eps: float=1e-08):
        super().__init__()
        self.voxel_size_x = int(grid_shape[0])
        self.voxel_size_y = int(grid_shape[1])
        self.voxel_size_z = int(grid_shape[2])
        self.eps = float(eps)
        self.padding_eps = float(padding_eps)
        bounds_min_tensor = torch.tensor(bounds_min, dtype=torch.float32)
        bounds_max_tensor = torch.tensor(bounds_max, dtype=torch.float32)
        if bounds_min_tensor.shape != (3,) or bounds_max_tensor.shape != (3,):
            raise ValueError('Invalid model input or configuration: ')
        if torch.any(bounds_max_tensor <= bounds_min_tensor):
            raise ValueError('Invalid model input or configuration: ')
        self.register_buffer('bounds_min', bounds_min_tensor)
        self.register_buffer('bounds_max', bounds_max_tensor)
        self.last_debug_info: Dict[str, torch.Tensor] | None = None

    def _build_point_mask(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[:, :3]
        finite_mask = torch.isfinite(xyz).all(dim=1)
        non_padding_mask = torch.any(torch.abs(points) > self.padding_eps, dim=1)
        return finite_mask & non_padding_mask

    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        if point_clouds.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(point_clouds.shape)}')
        (batch_size, seq_len, num_points, feat_dim) = point_clouds.shape
        if feat_dim < 3:
            raise ValueError(f'Invalid model input or configuration: {feat_dim}')
        device = point_clouds.device
        dtype = point_clouds.dtype
        flat_size = self.voxel_size_x * self.voxel_size_y * self.voxel_size_z
        bounds_min = self.bounds_min.to(device=device, dtype=dtype)
        bounds_max = self.bounds_max.to(device=device, dtype=dtype)
        span = (bounds_max - bounds_min).clamp_min(self.eps)
        flat_points = point_clouds.reshape(batch_size * seq_len * num_points, feat_dim)
        valid_mask = self._build_point_mask(flat_points)
        valid_points = flat_points[valid_mask]
        frame_ids = torch.arange(batch_size * seq_len, device=device, dtype=torch.long)
        frame_ids = frame_ids.repeat_interleave(num_points)[valid_mask]
        frame_valid_counts = valid_mask.reshape(batch_size, seq_len, num_points).sum(dim=2)
        valid_xyz = valid_points[:, :3]
        normalized_xyz = (valid_xyz - bounds_min.unsqueeze(0)) / span.unsqueeze(0)
        normalized_xyz = normalized_xyz.clamp(0.0, 1.0)
        x_idx = torch.floor(normalized_xyz[:, 0] * self.voxel_size_x).long().clamp(0, self.voxel_size_x - 1)
        y_idx = torch.floor(normalized_xyz[:, 1] * self.voxel_size_y).long().clamp(0, self.voxel_size_y - 1)
        z_idx = torch.floor(normalized_xyz[:, 2] * self.voxel_size_z).long().clamp(0, self.voxel_size_z - 1)
        voxel_idx = x_idx * (self.voxel_size_y * self.voxel_size_z) + y_idx * self.voxel_size_z + z_idx
        global_idx = frame_ids * flat_size + voxel_idx
        if feat_dim >= 4:
            doppler = valid_points[:, 3]
        else:
            doppler = torch.zeros_like(valid_points[:, 0])
        if feat_dim >= 5:
            snr = valid_points[:, 4]
        else:
            snr = torch.zeros_like(valid_points[:, 0])
        doppler = torch.nan_to_num(doppler, nan=0.0, posinf=0.0, neginf=0.0)
        snr = torch.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
        total_flat_size = batch_size * seq_len * flat_size
        flat_count = point_clouds.new_zeros(total_flat_size, dtype=dtype)
        flat_doppler = point_clouds.new_zeros(total_flat_size, dtype=dtype)
        flat_snr = point_clouds.new_zeros(total_flat_size, dtype=dtype)
        flat_count.index_add_(0, global_idx, torch.ones_like(doppler))
        flat_doppler.index_add_(0, global_idx, doppler)
        flat_snr.index_add_(0, global_idx, snr)
        voxel_shape = (batch_size, seq_len, self.voxel_size_x, self.voxel_size_y, self.voxel_size_z)
        voxel_count = flat_count.view(voxel_shape)
        voxel_doppler_sum = flat_doppler.view(voxel_shape)
        voxel_snr_sum = flat_snr.view(voxel_shape)
        mean_doppler = voxel_doppler_sum / (voxel_count + self.eps)
        mean_snr = voxel_snr_sum / (voxel_count + self.eps)
        voxels = torch.stack([mean_doppler, mean_snr], dim=-1)
        self.last_debug_info = {'bounds_min': bounds_min.detach(), 'bounds_max': bounds_max.detach(), 'frame_valid_counts': frame_valid_counts.detach(), 'channel_names': ['doppler', 'snr']}
        return voxels

class FeatureExtractionModule(nn.Module):

    def __init__(self, num_bones: int, num_rot_joints: int, gru_hidden: int=256, dropout: float=0.2):
        super().__init__()
        self.num_bones = int(num_bones)
        self.num_rot_joints = int(num_rot_joints)
        self.conv3d = nn.Sequential(nn.Conv3d(2, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2, stride=2), nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2, stride=2), nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d((2, 2, 2)))
        frame_feat_dim = 128 * 2 * 2 * 2
        self.temporal_gru = nn.GRU(input_size=frame_feat_dim, hidden_size=int(gru_hidden), num_layers=1, batch_first=True)
        self.shared_head = nn.Sequential(nn.Linear(int(gru_hidden), int(gru_hidden)), nn.ReLU(inplace=True), nn.Dropout(float(dropout)))
        self.bone_length_head = nn.Linear(int(gru_hidden), self.num_bones)
        self.rotation_head = nn.Linear(int(gru_hidden), self.num_rot_joints * 3)
        self.root_head = nn.Linear(int(gru_hidden), 3)

    def forward(self, voxels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if voxels.dim() != 6:
            raise ValueError(f'Invalid model input or configuration: {tuple(voxels.shape)}')
        (batch_size, seq_len, vx, vy, vz, channels) = voxels.shape
        if channels != 2:
            raise ValueError(f'Invalid model input or configuration: {channels}')
        x = voxels.permute(0, 1, 5, 2, 3, 4).contiguous().reshape(batch_size * seq_len, 2, vx, vy, vz)
        x = self.conv3d(x)
        x = x.flatten(start_dim=1)
        x = x.view(batch_size, seq_len, -1)
        (temporal_feat, _) = self.temporal_gru(x)
        last_feat = temporal_feat[:, -1, :]
        shared_feat = self.shared_head(last_feat)
        bone_lengths = F.softplus(self.bone_length_head(shared_feat)) + 0.0001
        rotations = torch.tanh(self.rotation_head(shared_feat)).view(batch_size, self.num_rot_joints, 3) * np.pi
        root_pos = self.root_head(shared_feat)
        return (bone_lengths, rotations, root_pos)

class FKLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_joints = len(JOINT_NAMES_17)
        self.parent_indices = list(PARENT_INDICES_17)
        self.child_order = list(FK_CHILD_ORDER_17)
        self.rotatable_joint_ids = list(ROTATABLE_JOINTS_17)
        self.num_rot_joints = len(self.rotatable_joint_ids)
        self.length_index_by_child = {child: idx for (idx, child) in enumerate(self.child_order)}
        self.rot_index_by_child = {child: idx for (idx, child) in enumerate(self.rotatable_joint_ids)}
        base_dirs = torch.tensor(BASE_DIRS_17, dtype=torch.float32)
        base_norm = torch.norm(base_dirs, dim=1, keepdim=True).clamp_min(1e-06)
        self.register_buffer('base_dirs', base_dirs / base_norm)

    @staticmethod
    def _batch_euler_to_matrix(angles: torch.Tensor) -> torch.Tensor:
        rx = angles[:, 0]
        ry = angles[:, 1]
        rz = angles[:, 2]
        (cosx, sinx) = (torch.cos(rx), torch.sin(rx))
        (cosy, siny) = (torch.cos(ry), torch.sin(ry))
        (cosz, sinz) = (torch.cos(rz), torch.sin(rz))
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        rx_mat = torch.stack([torch.stack([ones, zeros, zeros], dim=1), torch.stack([zeros, cosx, -sinx], dim=1), torch.stack([zeros, sinx, cosx], dim=1)], dim=1)
        ry_mat = torch.stack([torch.stack([cosy, zeros, siny], dim=1), torch.stack([zeros, ones, zeros], dim=1), torch.stack([-siny, zeros, cosy], dim=1)], dim=1)
        rz_mat = torch.stack([torch.stack([cosz, -sinz, zeros], dim=1), torch.stack([sinz, cosz, zeros], dim=1), torch.stack([zeros, zeros, ones], dim=1)], dim=1)
        return torch.bmm(torch.bmm(rz_mat, rx_mat), ry_mat)

    def compute_bone_lengths(self, joints: torch.Tensor) -> torch.Tensor:
        if joints.dim() != 3 or joints.shape[1] != self.num_joints or joints.shape[2] != 3:
            raise ValueError(f'Invalid model input or configuration: {self.num_joints}Invalid model input or configuration: {tuple(joints.shape)}')
        lengths = []
        for child_id in self.child_order:
            parent_id = self.parent_indices[child_id]
            lengths.append(torch.norm(joints[:, child_id, :] - joints[:, parent_id, :], dim=1))
        return torch.stack(lengths, dim=1)

    def forward(self, bone_lengths: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        if bone_lengths.dim() != 2:
            raise ValueError(f'Invalid model input or configuration: {tuple(bone_lengths.shape)}')
        if rotations.dim() != 3:
            raise ValueError(f'Invalid model input or configuration: {tuple(rotations.shape)}')
        batch_size = bone_lengths.shape[0]
        device = bone_lengths.device
        joints = torch.zeros((batch_size, self.num_joints, 3), device=device, dtype=bone_lengths.dtype)
        for child_id in self.child_order:
            parent_id = self.parent_indices[child_id]
            length_idx = self.length_index_by_child[child_id]
            bone_len = bone_lengths[:, length_idx].unsqueeze(1)
            local_dir = self.base_dirs[child_id].to(device=device, dtype=bone_lengths.dtype).unsqueeze(0).expand(batch_size, 3)
            local_offset = local_dir * bone_len
            if child_id in self.rot_index_by_child:
                rot_idx = self.rot_index_by_child[child_id]
                rot_angles = rotations[:, rot_idx, :]
                rot_mat = self._batch_euler_to_matrix(rot_angles)
                local_offset = torch.bmm(rot_mat, local_offset.unsqueeze(2)).squeeze(2)
            joints[:, child_id, :] = joints[:, parent_id, :] + local_offset
        return joints

class MMPoseFK(KeypointModel):
    supported_joints = tuple(POSE17_FROM_19)

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info: Dict | None=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info.get('num_classes', None)
        if self._num_classes is not None:
            raise ValueError('Invalid model input or configuration: ')
        self._num_points = int(info.get('num_keypoints', 19))
        if self._num_points != 19:
            raise ValueError(f'Invalid model input or configuration: {self._num_points}')
        self._seq_len = int(info.get('seq_len', 5))
        self._max_points = int(info.get('max_points', 300))
        self._model_input_dim = 5
        self._fk_num_points = len(JOINT_NAMES_17)
        self.training_primary_metric_only = True
        self.metric_joint_names = list(JOINT_NAMES_17)
        self.metric_parent_indices = list(PARENT_INDICES_17)
        self._root_loss_weight = float(info.get('fk_root_loss_weight', 0.5))
        self._length_loss_weight = float(info.get('fk_length_loss_weight', 1.0))
        grid_shape = tuple(info.get('voxel_grid_shape', (32, 32, 64)))
        if len(grid_shape) != 3:
            raise ValueError(f'Invalid model input or configuration: {grid_shape}')
        bounds_min = tuple(info.get('voxel_bounds_min', DEFAULT_VOXEL_BOUNDS_MIN))
        bounds_max = tuple(info.get('voxel_bounds_max', DEFAULT_VOXEL_BOUNDS_MAX))
        self.register_buffer('pose17_from_19', torch.tensor(POSE17_FROM_19, dtype=torch.long))
        self.register_buffer('foot_offsets', torch.tensor(FOOT_OFFSETS_FROM_ANKLE, dtype=torch.float32))
        self.register_buffer('position_loss_weights_17', torch.tensor([1.0, 1.0, 1.2, 1.2, 1.0, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.2, 1.0], dtype=torch.float32))
        self.voxelization = RadarPointCloudVoxelization(grid_shape=(int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])), bounds_min=(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])), bounds_max=(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])))
        self.fk_layer = FKLayer()
        self.feature_extractor = FeatureExtractionModule(num_bones=len(self.fk_layer.child_order), num_rot_joints=self.fk_layer.num_rot_joints, gru_hidden=int(info.get('gru_hidden', 256)), dropout=float(info.get('dropout', 0.2)))

    def _align_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1).repeat(1, self._seq_len, 1, 1)
        elif x.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        (batch_size, seq_len, num_points, feat_dim) = x.shape
        if seq_len < self._seq_len:
            pad_frames = x[:, -1:, :, :].repeat(1, self._seq_len - seq_len, 1, 1)
            x = torch.cat([x, pad_frames], dim=1)
        elif seq_len > self._seq_len:
            x = x[:, :self._seq_len, :, :]
        if num_points < self._max_points:
            pad_points = x.new_zeros((batch_size, x.shape[1], self._max_points - num_points, feat_dim))
            x = torch.cat([x, pad_points], dim=2)
        elif num_points > self._max_points:
            x = x[:, :, :self._max_points, :]
        if feat_dim < self._model_input_dim:
            pad_feats = x.new_zeros((batch_size, x.shape[1], x.shape[2], self._model_input_dim - feat_dim))
            x = torch.cat([x, pad_feats], dim=3)
        elif feat_dim > self._model_input_dim:
            x = x[:, :, :, :self._model_input_dim]
        return x.contiguous()

    def _pose19_to_fk17(self, pose_19: torch.Tensor) -> torch.Tensor:
        if pose_19.dim() == 2:
            return pose_19.index_select(0, self.pose17_from_19.to(device=pose_19.device))
        if pose_19.dim() == 3:
            return pose_19.index_select(1, self.pose17_from_19.to(device=pose_19.device))
        raise ValueError(f'Invalid model input or configuration: {tuple(pose_19.shape)}')

    def _expand_fk17_to_pose19(self, pose_17: torch.Tensor) -> torch.Tensor:
        if pose_17.dim() != 3 or pose_17.shape[1] != self._fk_num_points or pose_17.shape[2] != 3:
            raise ValueError(f'Invalid model input or configuration: {self._fk_num_points}Invalid model input or configuration: {tuple(pose_17.shape)}')
        batch_size = pose_17.shape[0]
        pose_19 = pose_17.new_zeros((batch_size, len(JOINT_NAMES_19), 3))
        for (out_idx, fk_idx) in POSE19_FROM_17.items():
            pose_19[:, out_idx, :] = pose_17[:, fk_idx, :]
        pose_19[:, 13, :] = pose_17[:, 3, :] + self.foot_offsets[0].to(device=pose_17.device, dtype=pose_17.dtype)
        pose_19[:, 17, :] = pose_17[:, 6, :] + self.foot_offsets[1].to(device=pose_17.device, dtype=pose_17.dtype)
        return pose_19

    def build_fk_targets(self, pose_target_19: torch.Tensor) -> Dict[str, torch.Tensor]:
        pose_target_17_abs = self._pose19_to_fk17(pose_target_19)
        root_target = pose_target_17_abs[:, 0, :]
        bone_length_target = self.fk_layer.compute_bone_lengths(pose_target_17_abs)
        return {'pose_target_17_abs': pose_target_17_abs, 'root_target': root_target, 'bone_length_target': bone_length_target}

    def compute_target_loss(self, model_outputs: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor], loss_fn: nn.Module) -> torch.Tensor:
        del loss_fn
        if 'pred_fk_abs' not in model_outputs:
            raise KeyError('Invalid model input or configuration: ')
        targets = self.build_fk_targets(batch_data['pose_target'])
        pred_fk_abs = model_outputs['pred_fk_abs']
        target_fk_abs = targets['pose_target_17_abs']
        per_joint_mse = (pred_fk_abs - target_fk_abs).pow(2).mean(dim=-1)
        joint_weights = self.position_loss_weights_17.to(device=pred_fk_abs.device, dtype=pred_fk_abs.dtype)
        weighted_position_loss = (per_joint_mse * joint_weights.unsqueeze(0)).mean()
        return weighted_position_loss

    def compute_auxiliary_losses(self, model_outputs: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor], loss_fn: nn.Module) -> Dict[str, torch.Tensor]:
        if 'pred_root' not in model_outputs or 'pred_bone_lengths' not in model_outputs:
            return {}
        targets = self.build_fk_targets(batch_data['pose_target'])
        root_loss = loss_fn(model_outputs['pred_root'], targets['root_target'])
        length_loss = loss_fn(model_outputs['pred_bone_lengths'], targets['bone_length_target'])
        return {'fk_root_loss': self._root_loss_weight * root_loss, 'fk_length_loss': self._length_loss_weight * length_loss}

    def compute_metric_inputs(self, model_outputs: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if 'pred_fk_abs' not in model_outputs:
            raise KeyError('Invalid model input or configuration: ')
        targets = self.build_fk_targets(batch_data['pose_target'])
        return (model_outputs['pred_fk_abs'], targets['pose_target_17_abs'])

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None=None, target_idx: torch.Tensor | None=None, return_dict: bool=False):
        raw_input = x
        x = self._align_input(x)
        voxel_data = self.voxelization(x)
        (bone_lengths, rotations, root_pos) = self.feature_extractor(voxel_data)
        joints_rel_17 = self.fk_layer(bone_lengths, rotations)
        joints_abs_17 = joints_rel_17 + root_pos.unsqueeze(1)
        joints_abs_19 = self._expand_fk17_to_pose19(joints_abs_17)
        if return_dict:
            return {'pred_seq': None, 'pred_target': joints_abs_17, 'pred_fk_abs': joints_abs_17, 'pred_fk_rel': joints_rel_17, 'pred_root': root_pos, 'pred_bone_lengths': bone_lengths, 'pred_visual_target': joints_abs_19, 'target_idx': target_idx, 'valid_mask': valid_mask}
        return joints_abs_19

def get_fk_joint_definition_17() -> Dict[str, List]:
    return {'joint_names': list(JOINT_NAMES_17), 'parent_indices': list(PARENT_INDICES_17), 'child_order': list(FK_CHILD_ORDER_17), 'rotatable_joint_ids': list(ROTATABLE_JOINTS_17)}
