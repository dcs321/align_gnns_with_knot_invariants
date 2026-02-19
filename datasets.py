import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.transforms import AddLaplacianEigenvectorPE

def create_hypergraph_dataset_from_pd(pd_notations, labels, node_feature_type="ones", embedding_used=False, use_uniform_edge_features=False, classification_or_regression="regression", number_of_period_in_circular=None, number_of_period_in_complex_circular=None):
    dataset = []
    max_num_of_nodes = 0
    for pd_notation, label in zip(pd_notations, labels):
        hypergraph_data, num_nodes = hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type, embedding_used, use_uniform_edge_features, classification_or_regression, number_of_period_in_circular, number_of_period_in_complex_circular)
        if num_nodes > max_num_of_nodes:
            max_num_of_nodes = num_nodes
        dataset.append(hypergraph_data)
    return dataset, max_num_of_nodes

def hypergraph_datapoint_from_pd(pd_notation, label, node_feature_type, embedding_used, use_uniform_edge_features, classification_or_regression, number_of_period_in_circular, number_of_period_in_complex_circular):
    num_nodes = np.array(pd_notation).max()
    
    if embedding_used:
        assert node_feature_type == "zeros" or node_feature_type == "numbers" or node_feature_type == "random_numbers" or node_feature_type == "numbers_with_random_circular_shift" or node_feature_type == "degree" or node_feature_type == "circular", "Now only zero, numbers, random_numbers, numbers_with_random_circular_shift, degree or circular node labels are supported."
    

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
    if classification_or_regression == "regression":
        y = torch.tensor(label, dtype=torch.float)
    elif classification_or_regression == "classification":
        y = torch.tensor(label, dtype=torch.long)
    
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    if node_feature_type == "zeros":
        node_features = torch.zeros((num_nodes,1), dtype=torch.float)
    elif node_feature_type == "ones":
        node_features = torch.ones((num_nodes,1), dtype=torch.float) 
    elif node_feature_type == "random":
        node_features = torch.randn((num_nodes, 16), dtype=torch.float)
    elif node_feature_type == "numbers":
        node_features = torch.arange(0, num_nodes, dtype=torch.float).reshape(-1,1)
    elif node_feature_type == "numbers_with_random_circular_shift":
        node_features = torch.arange(0, num_nodes, dtype=torch.long)
        shift = torch.randint(0, num_nodes, (1,)).item()
        node_features = (node_features + shift) % num_nodes
        node_features = node_features.to(torch.float).reshape(-1,1)
    elif node_feature_type == "random_numbers":
        node_features = torch.randperm(num_nodes).to(torch.float).reshape(-1, 1)
    elif node_feature_type == "laplacian":
        normal_edge_indices, normal_edge_weights = hypergraph_to_graph(hypergraph_edges)
        data_normal_graph = Data(edge_index=normal_edge_indices, edge_weight=normal_edge_weights, num_nodes=num_nodes)
        graph_laplacian_transformation = AddLaplacianEigenvectorPE(k=min(16, num_nodes-1), attr_name='x', is_undirected=True)
        node_features = graph_laplacian_transformation(data_normal_graph).x
        if node_features.shape[1] < 16:
            node_features = torch.cat([node_features, torch.zeros((num_nodes, 16 - node_features.shape[1]), dtype=torch.float)], dim=1)
    elif node_feature_type == "degree":
        normal_edge_indices, normal_edge_weights = hypergraph_to_graph(hypergraph_edges)
        degree = torch.zeros(num_nodes, dtype=torch.float)
        for i in range(normal_edge_indices.shape[1]):
            edge_begin = normal_edge_indices[0, i]
            degree[edge_begin] += normal_edge_weights[i]
        node_features = degree.reshape(-1,1)
    elif node_feature_type == "complex_circular":
        if number_of_period_in_complex_circular is None:
            theta = 2 * torch.pi * torch.arange(0, num_nodes, dtype=torch.float) / num_nodes
        else:
            theta = 2 * torch.pi * torch.arange(0, num_nodes, dtype=torch.float) / number_of_period_in_complex_circular
        cos_features = torch.cos(theta).reshape(-1,1)
        sin_features = torch.sin(theta).reshape(-1,1)
        node_features = torch.cat([cos_features, sin_features], dim=1)
    elif node_feature_type == "circular":
        if number_of_period_in_circular is None:
            circular_features = torch.arange(0, num_nodes, dtype=torch.long)
        else:
            circular_features = torch.arange(0, num_nodes, dtype=torch.long) % number_of_period_in_circular
        node_features = circular_features.reshape(-1,1).to(torch.float)
    else:
        raise NotImplementedError(f"Not implemented node feature type: {node_feature_type}")

    if embedding_used:
        node_features = node_features.to(torch.long)
        edge_features = edge_features.to(torch.long)
    

    return Data(x=node_features, edge_attr=edge_features, hyperedge_index=hypergraph_edges, y=y), num_nodes

def hypergraph_to_graph(hypergraph_edges):
    hyperedge_dict = dict()
    edge_dict = dict()
    
    hyperedge_nodes = hypergraph_edges[0].tolist()
    hyperedge_indices = hypergraph_edges[1].tolist()

    for node, hyperedge in zip(hyperedge_nodes, hyperedge_indices):
        if hyperedge not in hyperedge_dict:
            hyperedge_dict[hyperedge] = list()
        hyperedge_dict[hyperedge].append(node)

    for hyperedge, nodes in hyperedge_dict.items():
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                first_node, second_node = (nodes[i], nodes[j])
                if (first_node, second_node) not in edge_dict and (second_node, first_node) not in edge_dict:
                    edge_dict[(first_node, second_node)] = 1
                    edge_dict[(second_node, first_node)] = 1
                else:
                    edge_dict[(first_node, second_node)] += 1
                    edge_dict[(second_node, first_node)] += 1
    
    edge_indices = [[],[]]
    edge_weights = []
    for (first_node, second_node), weight in edge_dict.items():
        edge_indices[0].append(first_node)
        edge_indices[1].append(second_node)
        edge_weights.append(weight)

    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return edge_indices, edge_weights