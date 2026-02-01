import torch
from torch_geometric.data import Data
import numpy as np

def create_hypergraph_dataset_from_pd(pd_notations, labels, node_feature_type=None):
    dataset = []
    for pd_notation, label in zip(pd_notations, labels):
        hypergraph_data = hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type)
        dataset.append(hypergraph_data)
    return dataset

def hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type):
    num_nodes = np.array(pd_notation).max()
    
    if node_feature_type == "ones":
        node_features = torch.ones((num_nodes,1)) 
    elif node_feature_type == "random":
        node_features = torch.randn((num_nodes, 16))
    else:
        raise NotImplementedError(f"Not implemented node feature type: {node_feature_type}")
    

    hypergraph_edges = [[],[]]
    edge_features = []

    for c_idx, crossing in enumerate(pd_notation):

        for e_idx, edge_idx in enumerate(crossing):
            hypergraph_edges[0].append(edge_idx-1)
            hypergraph_edges[1].append(2*c_idx)
            edge_features.append(0)
            if e_idx % 2 == 1:
                hypergraph_edges[0].append(edge_idx-1)
                hypergraph_edges[1].append(2*c_idx+1)
                edge_features.append(1)

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long)
    y = torch.tensor(label, dtype=torch.float)
    edge_features = torch.tensor(edge_features, dtype=torch.long)

    return Data(x=node_features, edge_attr=edge_features, hyperedge_index=hypergraph_edges, y=y)