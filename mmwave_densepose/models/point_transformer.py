import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn, knn_graph
from torch_geometric.utils import scatter
from .base import KeypointModel
from .pointnet2 import resolve_point_input_dims, split_batched_point_inputs

class TransformerBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)
        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x

class TransitionDown(torch.nn.Module):

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce='max')
        (sub_pos, out) = (pos[id_clusters], x_out)
        return (out, sub_pos, sub_batch)

class PointTransformer(KeypointModel):

    def __init__(self, info=None, dim_model=[32, 64, 128, 256, 512], k=16):
        super().__init__(info)
        self.k = k
        self._sample_dim = int(info.get('sample_dim', info.get('feature_dim', 3)))
        (_, self._input_feature_dim, _) = resolve_point_input_dims(sample_dim=self._sample_dim, use_xyz_as_attr_when_sample_dim3=True)
        in_channels = self._input_feature_dim
        self._num_classes = info['num_classes']
        out_channels = self._num_classes if self._num_classes is not None else info['num_keypoints']
        in_channels = max(in_channels, 1)
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_down.append(TransitionDown(in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k))
            self.transformers_down.append(TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))
        if self._num_classes is None:
            num_points = info['num_keypoints']
            self._num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([dim_model[-1], 64, 3], norm=None)
            self.mlp_output = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, data):
        if data.dim() == 4:
            data = data[:, -1].contiguous()
        if data.dim() != 3:
            raise ValueError(f'PointTransformer expects input shape [B, N, C>=3], got {tuple(data.shape)}')
        batchsize = data.shape[0]
        npoints = data.shape[1]
        (x, pos) = split_batched_point_inputs(data, sample_dim=self._sample_dim, use_xyz_as_attr_when_sample_dim3=True)
        x = x.reshape((batchsize * npoints, self._input_feature_dim))
        pos = pos.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)
        for i in range(len(self.transformers_down)):
            (x, pos, batch) = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)
        x = global_mean_pool(x, batch)
        if self._num_classes is None:
            y = []
            for i in range(self._num_points):
                y.append(self.mlp_output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
            return y
        out = self.mlp_output(x)
        return out
