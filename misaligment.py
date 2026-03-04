import torch
import torch.nn.functional as F

def compute_misaligment_score(dataset, max_num_of_nodes, node_feature_type, regression_or_classification):
    outputs = []
    targets = []
    labels = []

    if regression_or_classification == "classification":
        num_of_categories = 0
        for data in dataset:
            label = data.y
            if label[0] > num_of_categories:
                num_of_categories = label[0]
            labels.append(label[0])
        num_of_categories = num_of_categories + 1

    for data in dataset:
        node_features, label, hyperedge_index = data.x, data.y, data.hyperedge_index

        hypergraph_nodes = hyperedge_index[0].clone().detach().to(torch.long)
        hypergraph_edges = hyperedge_index[1].clone().detach().to(torch.long)   
       
        num_nodes = hypergraph_nodes.max().item() + 1
        num_edges = hypergraph_edges.max().item() + 1


        if node_feature_type == "numbers":
            node_features = F.one_hot(node_features.squeeze(), num_classes=max_num_of_nodes).to(torch.float)
       


        incidency_matrix = torch.zeros((num_nodes, num_edges), dtype=torch.float, device=hypergraph_nodes.device)
        incidency_matrix[hypergraph_nodes, hypergraph_edges] = 1.0

        edge_degrees = incidency_matrix.sum(dim=0)+ 1e-6
        node_degrees = incidency_matrix.sum(dim=1)+ 1e-6

        node_degrees_inv_sqrt = torch.diag(torch.pow(node_degrees, -0.5))
        edge_degrees_inv = torch.diag(torch.pow(edge_degrees, -1.0))


        hypergraph_convolution_operator = node_degrees_inv_sqrt @ incidency_matrix @ edge_degrees_inv @ incidency_matrix.T @ node_degrees_inv_sqrt
 
        output_features = hypergraph_convolution_operator @ node_features
        pooled_output_features = torch.cat([output_features.mean(dim=0), output_features.sum(dim=0), output_features.max(dim=0)[0]])
        
        if regression_or_classification == "regression":
            targets.append(label[0])
        elif regression_or_classification == "classification":
            targets.append(F.one_hot(label[0], num_classes=num_of_categories).to(torch.float))
        outputs.append(pooled_output_features)
    
    outputs = torch.stack(outputs)
    targets = torch.stack(targets)
    labels = torch.tensor(labels)

    if regression_or_classification == "regression":
        normalized_misaligment_score = compute_regression_misaligment_score(outputs,targets)
        return normalized_misaligment_score
    elif regression_or_classification == "classification":
        normalized_misaligment_score, normalized_misaligment_score_weighted = compute_classification_misaligment_score(labels,outputs,targets)
        return normalized_misaligment_score, normalized_misaligment_score_weighted
    

def compute_regression_misaligment_score(outputs, targets):

    targets_centered = targets - targets.mean(dim=0,keepdim=True)
    variance = torch.sum(targets_centered ** 2)
    
    w = torch.linalg.lstsq(outputs, targets_centered, rcond=None).solution

    difference = targets_centered - outputs @ w

    misaligment_score = torch.sum(difference ** 2) / variance

    return torch.clamp(misaligment_score, min=0.0, max=1.0).item()

def compute_classification_misaligment_score(labels, outputs, targets):
    num_samples = targets.shape[0]
    unique_categories, counts = torch.unique(labels, return_counts=True)
    category_ratios = counts.to(torch.float) / num_samples

    weighted_misaligment_score = 0.0
    unweighted_misaligment_score = 0.0
    for i, category in enumerate(unique_categories):
        target_mask = (labels == category).to(torch.float)
        category_missaligment = compute_regression_misaligment_score(outputs,target_mask)
        weighted_misaligment_score = weighted_misaligment_score + category_missaligment*category_ratios[i].item()
        unweighted_misaligment_score = unweighted_misaligment_score + category_missaligment
    
    unweighted_misaligment_score = unweighted_misaligment_score / len(unique_categories)
    return unweighted_misaligment_score, weighted_misaligment_score


