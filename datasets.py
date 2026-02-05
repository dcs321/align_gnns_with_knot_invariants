import torch
from torch_geometric.data import Data
import numpy as np

def create_hypergraph_dataset_from_pd(pd_notations, labels, node_feature_type="ones", embedding_used=False, use_uniform_edge_features=False):
    dataset = []
    max_num_of_nodes = 0
    for pd_notation, label in zip(pd_notations, labels):
        hypergraph_data, num_nodes = hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type, embedding_used, use_uniform_edge_features)
        if num_nodes > max_num_of_nodes:
            max_num_of_nodes = num_nodes
        dataset.append(hypergraph_data)
    return dataset, max_num_of_nodes

def hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type, embedding_used, use_uniform_edge_features):
    num_nodes = np.array(pd_notation).max()
    
    if embedding_used:
        assert node_feature_type == "zeros" or node_feature_type == "numbers", "Now only zero or numbers node label is supported."

    if node_feature_type == "zeros":
        node_features = torch.zeros((num_nodes,1), dtype=torch.float)
    elif node_feature_type == "ones":
        node_features = torch.ones((num_nodes,1), dtype=torch.float) 
    elif node_feature_type == "random":
        node_features = torch.randn((num_nodes, 16), dtype=torch.float)
    elif node_feature_type == "numbers":
        node_features = torch.arange(0, num_nodes, dtype=torch.float).reshape(-1,1)
    else:
        raise NotImplementedError(f"Not implemented node feature type: {node_feature_type}")
    

    hypergraph_edges = [[],[]]
    edge_features = []

    for c_idx, crossing in enumerate(pd_notation):

        for e_idx, edge_idx in enumerate(crossing):
            hypergraph_edges[0].append(edge_idx-1)
            hypergraph_edges[1].append(2*c_idx)
            if e_idx % 2 == 1:
                hypergraph_edges[0].append(edge_idx-1)
                hypergraph_edges[1].append(2*c_idx+1)
        if not use_uniform_edge_features:
            edge_features.append([0]) #crossing
            edge_features.append([1]) #over
        else:
            edge_features.append([0]) 
            edge_features.append([0])
    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long)
    y = torch.tensor(label, dtype=torch.float)
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    if embedding_used:
        node_features = node_features.to(torch.long)
        edge_features = edge_features.to(torch.long)
    

    return Data(x=node_features, edge_attr=edge_features, hyperedge_index=hypergraph_edges, y=y), num_nodes