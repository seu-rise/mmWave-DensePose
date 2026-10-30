import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import KeypointModel

class EdgeConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def knn(self, x, k=6):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise = -xx - inner - xx.transpose(2, 1)
        idx = pairwise.topk(k=k, dim=-1)[1]
        return idx

    def forward(self, x):
        (B, C, N) = x.shape
        idx = self.knn(x, k=6)
        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        x_t = x.transpose(2, 1).contiguous()
        neighbor = x_t.view(B * N, -1)[idx].view(B, N, 6, C)
        x_t = x_t.view(B, N, 1, C).repeat(1, 1, 6, 1)
        edge = torch.cat([x_t, neighbor - x_t], dim=-1)
        out = self.mlp(edge).max(dim=2)[0]
        return out.transpose(2, 1)

class ChannelAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))

    def forward(self, x):
        (B, C, N) = x.shape
        avg = x.mean(dim=-1)
        maxp = x.max(dim=-1)[0]
        s = torch.sigmoid(self.mlp(avg) + self.mlp(maxp)).view(B, C, 1)
        return x * s

class AEDGCNN(nn.Module):

    def __init__(self, in_dim=5):
        super().__init__()
        self.conv1 = EdgeConv(in_dim, 64)
        self.at1 = ChannelAttention(64)
        self.conv2 = EdgeConv(64, 128)
        self.at2 = ChannelAttention(128)
        self.conv3 = EdgeConv(128, 256)
        self.at3 = ChannelAttention(256)

    def forward(self, x):
        x1 = self.at1(self.conv1(x))
        x2 = self.at2(self.conv2(x1))
        x3 = self.at3(self.conv3(x2))
        concat = torch.cat([x1, x2, x3], dim=1)
        global_feat = torch.max(concat, dim=-1)[0]
        return (global_feat, concat)

class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, H, A_hat):
        H = self.relu(self.W(A_hat @ H))
        return H

class SkeletonGCN(nn.Module):

    def __init__(self, in_dim, hid_dim, num_joints=19):
        super().__init__()
        self.num_joints = num_joints
        self.A = nn.Parameter(torch.eye(num_joints))
        self.gcn1 = GraphConv(in_dim, hid_dim)
        self.gcn2 = GraphConv(hid_dim, hid_dim)

    def forward(self, H):
        A = self.A + torch.eye(self.num_joints, device=H.device)
        D = torch.diag_embed(torch.sum(A, dim=-1) ** (-0.5))
        A_hat = D @ A @ D
        H = self.gcn1(H, A_hat)
        H = self.gcn2(H, A_hat)
        return H

class PosePrior(nn.Module):

    def __init__(self, in_dim, out_dim, num_joints=19):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))
        self.M = num_joints

    def forward(self, g):
        feat = self.mlp(g).unsqueeze(1).repeat(1, self.M, 1)
        return feat

class MGPH(nn.Module):

    def __init__(self, num_joints=19, gcn_dim=64, prior_dim=64):
        super().__init__()
        self.num_joints = int(num_joints)
        self.gcn = SkeletonGCN(448, gcn_dim, num_joints)
        self.prior = PosePrior(448, prior_dim, num_joints)
        self.head = nn.Sequential(nn.Linear(gcn_dim + prior_dim, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, global_feat):
        B = global_feat.shape[0]
        H = global_feat.unsqueeze(1).repeat(1, self.num_joints, 1)
        gcn_feat = self.gcn(H)
        prior_feat = self.prior(global_feat)
        fuse = torch.cat([gcn_feat, prior_feat], dim=-1)
        return self.head(fuse)

class _PoseGraphNetCore(nn.Module):

    def __init__(self, in_dim=5, num_joints=19):
        super().__init__()
        self.backbone = AEDGCNN(in_dim=in_dim)
        self.mgph = MGPH(num_joints=num_joints)

    def forward(self, points):
        (g_feat, _) = self.backbone(points)
        joints = self.mgph(g_feat)
        return joints

class PoseGraphNet(KeypointModel):

    @property
    def task_type(self) -> str:
        return 'keypoint'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info.get('num_classes')
        if self._num_classes is not None:
            raise ValueError('Invalid model input or configuration: ')
        self._num_points = int(info.get('num_keypoints', 19))
        self._max_points = int(info.get('max_points', 300))
        self._model_input_dim = 5
        self.core = _PoseGraphNetCore(in_dim=self._model_input_dim, num_joints=self._num_points)

    def _align_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        x = x.transpose(1, 2).contiguous()
        (bsz, c, n) = x.shape
        if c < self._model_input_dim:
            pad = x.new_zeros((bsz, self._model_input_dim - c, n))
            x = torch.cat([x, pad], dim=1)
        elif c > self._model_input_dim:
            x = x[:, :self._model_input_dim, :]
        if n < self._max_points:
            pad = x.new_zeros((bsz, x.size(1), self._max_points - n))
            x = torch.cat([x, pad], dim=2)
        elif n > self._max_points:
            x = x[:, :, :self._max_points]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x[:, -1].contiguous()
        x = self._align_input(x)
        return self.core(x)
