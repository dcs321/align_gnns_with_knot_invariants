
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import HypergraphConv, global_mean_pool, global_max_pool, global_add_pool


class HyperGNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.graph_conv1 = HypergraphConv(input_dims, hidden_dims)
        self.graph_conv2 = HypergraphConv(hidden_dims, hidden_dims) 
        self.relu = ReLU()
        self.linear = Linear(3*hidden_dims, output_dims)

    def forward(self, data):
        node_features, hyperedge, batch = data.x, data.hyperedge_index, data.batch

        x = self.graph_conv1(node_features, hyperedge)
        x = self.relu(x)
        x_2 = self.graph_conv2(x, hyperedge)
        x_2 = self.relu(x_2)
        x = x + x_2

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        x = torch.cat([x_mean, x_max, x_sum], dim=1)

        return self.linear(x)