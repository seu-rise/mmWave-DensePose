import torch
from torch_geometric.nn import MLP
from .base import KeypointModel
from .pointnet2 import PointNetBackbone

class SelfAttentionBlock(torch.nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim * 4), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim * 4, hidden_dim))
        self.ffn_dropout = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (attn_out, _) = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_out))
        return x

class SkeletonAwarePartAttentionBlock(torch.nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_part_tokens: int=6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_part_tokens = int(num_part_tokens)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim))
        self.ffn_dropout = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.LayerNorm(self.hidden_dim)
        bias = self._init_skeleton_bias(self.num_part_tokens)
        self.skeleton_bias = torch.nn.Parameter(bias)

    @staticmethod
    def _init_skeleton_bias(num_part_tokens: int) -> torch.Tensor:
        bias = torch.zeros(num_part_tokens, num_part_tokens)
        if num_part_tokens != 6:
            return bias
        bias.fill_(-0.05)
        bias.fill_diagonal_(0.05)
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 3), (4, 5)]
        for (i, j) in edges:
            bias[i, j] = 0.05
            bias[j, i] = 0.05
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.skeleton_bias.to(device=x.device, dtype=x.dtype)
        (attn_out, _) = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_out))
        return x

class PASTNet(KeypointModel):

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info['num_classes']
        self._num_points = info['num_keypoints']
        self._seq_len = int(info.get('seq_len', 5))
        self._sample_dim = int(info.get('sample_dim', info.get('feature_dim', 3)))
        self._radar_point_size = int(info.get('radar_point_size', info.get('max_points', 300)))
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
        self.spatial_blocks = torch.nn.ModuleList([SkeletonAwarePartAttentionBlock(hidden_dim=self._part_dim, num_heads=self._part_heads, dropout=self._dropout, num_part_tokens=self._num_part_tokens) for _ in range(self._spatial_layers)])
        self.temporal_blocks = torch.nn.ModuleList([SelfAttentionBlock(hidden_dim=self._part_dim, num_heads=self._temporal_heads, dropout=self._dropout) for _ in range(self._temporal_layers)])
        fusion_dim = 1024 + self._num_part_tokens * self._part_dim
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

    def _build_part_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        (batch_size, seq_len, _) = frame_features.shape
        tokens = self.part_generator(frame_features)
        tokens = tokens.reshape(batch_size, seq_len, self._num_part_tokens, self._part_dim)
        tokens = tokens + self.part_type_embedding
        tokens = tokens.reshape(batch_size * seq_len, self._num_part_tokens, self._part_dim)
        for block in self.spatial_blocks:
            tokens = block(tokens)
        tokens = tokens.reshape(batch_size, seq_len, self._num_part_tokens, self._part_dim)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        tokens = tokens.reshape(batch_size * self._num_part_tokens, seq_len, self._part_dim)
        for block in self.temporal_blocks:
            tokens = block(tokens)
        tokens = tokens.reshape(batch_size, self._num_part_tokens, seq_len, self._part_dim)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        return tokens

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            data = data.unsqueeze(1)
        if data.dim() != 4:
            raise ValueError(f'PASTNet expects input shape [B, T, N, C], got {tuple(data.shape)}')
        frame_features = self._encode_frame_sequence(data)
        part_tokens = self._build_part_tokens(frame_features)
        current_global = frame_features[:, -1, :]
        current_part = part_tokens[:, -1, :, :].reshape(data.size(0), -1)
        fused_feat = torch.cat([current_global, current_part], dim=-1)
        return self._decode_target(fused_feat)
