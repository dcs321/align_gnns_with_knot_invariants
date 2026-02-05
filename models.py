
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Embedding
from torch_geometric.nn import HypergraphConv, global_mean_pool, global_max_pool, global_add_pool


class HyperGNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, use_attention = False, number_of_attention_heads=1, type_of_attention="node", embedding_used=False, max_num_of_nodes=None):
        super().__init__()
        self.use_attention = use_attention
        self.embedding_used = embedding_used

        if self.embedding_used:
            self.edge_embedding = Embedding(2, hidden_dims)
            if max_num_of_nodes is not None:
                self.node_embedding = Embedding(max_num_of_nodes, hidden_dims)
            else:
                self.node_embedding = Embedding(1, hidden_dims)
            input_dims = hidden_dims

        if use_attention:
            assert hidden_dims % number_of_attention_heads == 0, "Hidden dimensions must be divisible by the number of attention heads."
            first_hidden_dims = hidden_dims // number_of_attention_heads
        else:
            first_hidden_dims = hidden_dims

        self.graph_conv1 = HypergraphConv(input_dims, first_hidden_dims, use_attention=use_attention, heads=number_of_attention_heads, attention_mode=type_of_attention)
        self.graph_conv2 = HypergraphConv(hidden_dims, hidden_dims)

        self.relu = ReLU()
        self.linear = Linear(3*hidden_dims, output_dims)

    def forward(self, data):
        node_features, hyperedge_indices, hyperedge_features, batch = data.x, data.hyperedge_index, data.edge_attr, data.batch

        if self.embedding_used:
            hyperedge_features = self.edge_embedding(hyperedge_features.flatten())
            node_features = self.node_embedding(node_features.flatten())

        if self.use_attention:
            x = self.graph_conv1(node_features, hyperedge_indices, hyperedge_attr=hyperedge_features)
        else:
            x = self.graph_conv1(node_features, hyperedge_indices)

        x = self.relu(x)
        
        x_2 = self.graph_conv2(x, hyperedge_indices)
        
        x = x + x_2
        x = self.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        x = torch.cat([x_mean, x_max, x_sum], dim=1)

        return self.linear(x)