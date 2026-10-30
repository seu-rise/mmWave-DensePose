import torch
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from .base import KeypointModel

class DGCNN(KeypointModel):

    def __init__(self, k=30, aggr='max', info=None):
        super().__init__(info)
        self._num_classes = info['num_classes']
        self._sample_dim = max(3, int(info.get('sample_dim', info.get('feature_dim', 3))))
        self.conv = torch.nn.ModuleList([])
        n = self._sample_dim
        conv_layer = (32, 32, 32)
        dense_layer = (1024, 1024, 256, 128)
        for layer in conv_layer:
            edgeconv = DynamicEdgeConv(MLP([n * 2, layer, layer, layer]), k, aggr)
            self.conv.append(edgeconv)
            n = layer
        self.lin1 = MLP([sum(conv_layer), dense_layer[0]])
        if self._num_classes is None:
            num_points = info['num_keypoints']
            self._num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([*dense_layer, 3], dropout=0.5, norm=None)
            self.output = torch.nn.ModuleDict(point_branches)
        else:
            self.output = MLP([*dense_layer, self._num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        if data.dim() == 4:
            data = data[:, -1].contiguous()
        if data.dim() != 3 or data.size(-1) < 3:
            raise ValueError(f'DGCNN expects input shape [B, N, C>=3], got {tuple(data.shape)}')
        batchsize = data.shape[0]
        npoints = data.shape[1]
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        feature_dim = data.size(-1)
        if feature_dim < self._sample_dim:
            pad = data.new_zeros(batchsize, npoints, self._sample_dim - feature_dim)
            data = torch.cat([data, pad], dim=-1)
        x = data[:, :, :self._sample_dim].reshape((batchsize * npoints, self._sample_dim))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        xs = []
        for conv in self.conv:
            x = conv(x, batch)
            xs.append(x)
        x4 = self.lin1(torch.cat(xs, dim=1))
        x5 = global_max_pool(x4, batch)
        if self._num_classes is None:
            y = []
            for i in range(self._num_points):
                y.append(self.output[f'branch_{i}'](x5))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x5)
        return y
