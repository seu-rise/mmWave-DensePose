import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import KeypointModel

class TNet(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        return x.view(B, self.k, self.k)

class PointNetEncoder(nn.Module):

    def __init__(self, input_dim=5):
        super().__init__()
        self.input_tnet = TNet(k=input_dim)
        self.feat_tnet = TNet(k=64)
        self.mlp1 = nn.Sequential(nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp4 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

    def forward(self, x):
        trans_input = self.input_tnet(x)
        x = torch.bmm(trans_input, x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        trans_feat = self.feat_tnet(x)
        x = torch.bmm(trans_feat, x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = torch.max(x, 2)[0]
        return (x, trans_input, trans_feat)

class PntPose(nn.Module):

    def __init__(self, input_dim=5, num_joints=19, dropout=0.2):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=input_dim)
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_joints * 3))
        self.num_joints = num_joints

    def forward(self, x):
        (feat, _, _) = self.encoder(x)
        out = self.mlp(feat)
        return out

class MHSA(nn.Module):

    def __init__(self, dim=72, heads=24):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        (attn_out, _) = self.attn(x, x, x)
        return self.norm(attn_out + x)

class _PntPoseATCore(nn.Module):

    def __init__(self, input_dim=5, num_joints=19, seq_len=50, lstm_hidden=72, attn_dim=72, attn_heads=24, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.pntpose = PntPose(input_dim=input_dim, num_joints=num_joints, dropout=0.0)
        self.lstm = nn.LSTM(3 * num_joints, lstm_hidden, batch_first=True)
        self.mhsa = MHSA(attn_dim, attn_heads)
        self.reg = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * attn_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_joints * 3))

    def extract_feat(self, x):
        feat = self.pntpose(x)
        return feat

    def forward(self, x_seq):
        (B, T, D, N) = x_seq.shape
        feats = []
        for t in range(T):
            feat = self.extract_feat(x_seq[:, t])
            feats.append(feat)
        feat_seq = torch.stack(feats, dim=1)
        (lstm_out, _) = self.lstm(feat_seq)
        attn_out = self.mhsa(lstm_out)
        pose = self.reg(attn_out)
        return pose.view(B, self.num_joints, 3)

class PntPoseAT(KeypointModel):

    @property
    def task_type(self) -> str:
        return 'keypoint_muti'

    def __init__(self, info=None):
        info = info or {}
        super().__init__(info)
        self._num_classes = info.get('num_classes')
        if self._num_classes is not None:
            raise ValueError('Invalid model input or configuration: ')
        self._num_points = int(info.get('num_keypoints', 19))
        self._seq_len = int(info.get('seq_len', 5))
        self._max_points = int(info.get('max_points', 300))
        self._model_input_dim = 5
        self.core = _PntPoseATCore(input_dim=self._model_input_dim, num_joints=self._num_points, seq_len=self._seq_len, lstm_hidden=int(info.get('lstm_hidden', 72)), attn_dim=int(info.get('attn_dim', 72)), attn_heads=int(info.get('attn_heads', 24)), dropout=float(info.get('dropout', 0.2)))

    def _align_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1).repeat(1, self._seq_len, 1, 1)
        elif x.dim() != 4:
            raise ValueError(f'Invalid model input or configuration: {tuple(x.shape)}')
        (B, T, N, C) = x.shape
        if T < self._seq_len:
            pad = x[:, -1:, :, :].repeat(1, self._seq_len - T, 1, 1)
            x = torch.cat([x, pad], dim=1)
        elif T > self._seq_len:
            x = x[:, :self._seq_len, :, :]
        if N < self._max_points:
            pad = x.new_zeros((x.size(0), x.size(1), self._max_points - N, x.size(3)))
            x = torch.cat([x, pad], dim=2)
        elif N > self._max_points:
            x = x[:, :, :self._max_points, :]
        if C < self._model_input_dim:
            pad = x.new_zeros((x.size(0), x.size(1), x.size(2), self._model_input_dim - C))
            x = torch.cat([x, pad], dim=3)
        elif C > self._model_input_dim:
            x = x[:, :, :, :self._model_input_dim]
        return x.permute(0, 1, 3, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._align_input(x)
        return self.core(x)
