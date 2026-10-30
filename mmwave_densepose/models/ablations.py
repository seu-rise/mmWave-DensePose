import torch
from torch_geometric.nn import MLP
from .base import KeypointModel
from .pointnet2 import PointNetBackbone
from .pastnet import SelfAttentionBlock, SkeletonAwarePartAttentionBlock

class _BasePartFactorizedAblation(KeypointModel):

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info=None, fusion_dim=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info['num_classes']
        self._num_points = info['num_keypoints']
        self._seq_len = int(info.get('seq_len', 5))
        self._sample_dim = int(info.get('sample_dim', info.get('feature_dim', 3)))
        self._num_part_tokens = int(info.get('num_part_tokens', 6))
        self._part_dim = int(info.get('part_dim', 128))
        self._part_heads = int(info.get('part_heads', 4))
        self._spatial_layers = int(info.get('spatial_layers', info.get('part_layers', 1)))
        self._temporal_layers = int(info.get('temporal_layers', 1))
        self._dropout = float(info.get('dropout', info.get('attn_dropout', 0.1)))
        self._temporal_heads = int(info.get('temporal_heads', self._part_heads))
        if self._part_dim % self._part_heads != 0:
            raise ValueError(f'part_dim ({self._part_dim}) must be divisible by part_heads ({self._part_heads}).')
        if self._part_dim % self._temporal_heads != 0:
            raise ValueError(f'part_dim ({self._part_dim}) must be divisible by temporal_heads ({self._temporal_heads}).')
        self.backbone = PointNetBackbone(sample_dim=self._sample_dim)
        self.part_generator = torch.nn.Linear(1024, self._num_part_tokens * self._part_dim)
        self.part_type_embedding = torch.nn.Parameter(torch.zeros(1, 1, self._num_part_tokens, self._part_dim))
        torch.nn.init.trunc_normal_(self.part_type_embedding, std=0.02)
        self._build_decoder(fusion_dim or 1024 + self._num_part_tokens * self._part_dim)

    def _build_decoder(self, fusion_dim: int) -> None:
        if self._num_classes is None:
            point_branches = {}
            for i in range(self._num_points):
                point_branches[f'branch_{i}'] = MLP([fusion_dim, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([fusion_dim, 512, 256, self._num_classes], dropout=0.5, norm=None)

    def _encode_frame_sequence(self, data: torch.Tensor) -> torch.Tensor:
        (batch_size, seq_len, num_points, feat_dim) = data.shape
        flattened = data.reshape(batch_size * seq_len, num_points, feat_dim)
        encoded = self.backbone(flattened)
        return encoded.reshape(batch_size, seq_len, -1)

    def _decode_target(self, fused_feat: torch.Tensor) -> torch.Tensor:
        if self._num_classes is None:
            outputs = []
            for i in range(self._num_points):
                outputs.append(self.mlp[f'branch_{i}'](fused_feat))
            return torch.stack(outputs, dim=1)
        return self.mlp(fused_feat)

    def _prepare_input(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            data = data.unsqueeze(1)
        if data.dim() != 4:
            raise ValueError(f'{self.__class__.__name__} expects input shape [B, T, N, C], got {tuple(data.shape)}')
        if self.use_single_frame_input:
            data = data[:, -1:].contiguous()
        return data

    def _generate_part_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        (batch_size, seq_len, _) = frame_features.shape
        tokens = self.part_generator(frame_features)
        tokens = tokens.reshape(batch_size, seq_len, self._num_part_tokens, self._part_dim)
        return tokens + self.part_type_embedding

    def _flatten_last_tokens(self, data: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[:, -1, :, :].reshape(data.size(0), -1)

    def _build_part_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self._prepare_input(data)
        frame_features = self._encode_frame_sequence(data)
        part_tokens = self._build_part_tokens(frame_features)
        current_global = frame_features[:, -1, :]
        current_part = self._flatten_last_tokens(data, part_tokens)
        fused_feat = torch.cat([current_global, current_part], dim=-1)
        return self._decode_target(fused_feat)

class _SpatialOnlyNet(_BasePartFactorizedAblation):
    use_single_frame_input = True

    def __init__(self, info=None):
        super().__init__(info=info)
        self.spatial_blocks = torch.nn.ModuleList([SkeletonAwarePartAttentionBlock(hidden_dim=self._part_dim, num_heads=self._part_heads, dropout=self._dropout, num_part_tokens=self._num_part_tokens) for _ in range(self._spatial_layers)])

    def _build_part_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        (batch_size, seq_len, _) = frame_features.shape
        tokens = self._generate_part_tokens(frame_features)
        tokens = tokens.reshape(batch_size * seq_len, self._num_part_tokens, self._part_dim)
        for block in self.spatial_blocks:
            tokens = block(tokens)
        return tokens.reshape(batch_size, seq_len, self._num_part_tokens, self._part_dim)
