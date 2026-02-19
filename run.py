import argparse
import pandas as pd
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, CrossEntropyLoss
import wandb

from data_parsers import parse_crossing_number, parse_volume, parse_pd_notation, parse_list_of_features
from datasets import create_hypergraph_dataset_from_pd
from models import HyperGNN
from training import training_loop
from evaluation import create_test_predictions_and_targets, compute_test_loss, compute_test_accuracy_from_mape, plot_predictions_vs_targets, compute_test_r2_score, compute_test_accuracy_for_classification, plot_confusion_matrix
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/knot_data_merged.csv" ,required=False, help='Dataset path')
    parser.add_argument('--model', type=str, default="hyper_gnn", required=False, help='Model type')
    parser.add_argument('--data_type', type=str, default="hyper_graph", required=False, help='Data type')
    parser.add_argument('--data_loader', type=str, default="graph", required=False, help='Task type')
    parser.add_argument('--notation', type=str, default="pd", required=False, help='Knot notation')
    parser.add_argument('--node_feature_type', type=str, default="ones", required=False, help='Node feature type')
    parser.add_argument('--target_invariant', type=str, default="volume", required=False, help='Target knot invariant')
    parser.add_argument('--train_split_ratio', type=float, default=0.8, required=False, help='Train split ratio')
    parser.add_argument('--validation_split_ratio', type=float, default=0.1, required=False, help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, required=False, help='Learning rate')
    parser.add_argument('--number_of_epochs', type=int, default=500, required=False, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, required=False, help='Early stopping patience')
    parser.add_argument('--model_save_path', type=str,default="models/model.pth", required=False, help='Model save path')
    parser.add_argument('--figure_save_path', type=str, default="figures/plot.png", required=False, help='Figure save path')
    parser.add_argument('--hidden_dims', type=int, default=64, required=False, help='Hidden dimensions')
    parser.add_argument('--device', type=str, default='cpu', required=False, help='Device (cpu or cuda)')
    parser.add_argument('--criterion', type=str, default='mse', required=False, help='Criterion')
    parser.add_argument('--optimizer', type=str, default='adam', required=False, help='Optimizer')
    parser.add_argument('--random_seed', type=int, default=42, required=False, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--run_name', type=str, default='experiment', required=False, help='Wandb run name')
    parser.add_argument('--use_attention_in_hypergraph', default=False, action='store_true', help='Use attention in HyperGNN')
    parser.add_argument('--number_of_attention_heads_in_hypergraph', type=int, default=1, required=False, help='Number of attention heads in HyperGNN')
    parser.add_argument('--type_of_attention_in_hypergraph', type=str, default="node", required=False, help='Type of attention in HyperGNN')
    parser.add_argument('--embedding_used', default=False, action='store_true', required=False, help='Use embedding for node and edge features.')
    parser.add_argument('--regression_or_classification', type=str, default="regression", required=False, help='Regression or classification task')
    parser.add_argument('--uniform_edge_features', action='store_true', required=False, help='Use uniform edge features.')
    parser.add_argument('--num_of_layers_in_hypergraph', type=int, default=2, required=False, help='Number of layers in HyperGNN')
    parser.add_argument('--number_of_period_in_circular', type=int, default=None, required=False, help='Number of period in circular node embedding')
    parser.add_argument('--number_of_period_in_complex_circular', type=int, default=None, required=False, help='Number of period in complex circular node embedding')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(name=args.run_name, config=vars(args), save_code=True)
        wandb.run.log_code(".")
 
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    df = pd.read_csv(args.data_path)

    if args.target_invariant == "volume":
        y = parse_list_of_features(list(df["Volume"]), parse_volume)
    elif args.target_invariant == "crossing_number":
        y = parse_list_of_features(list(df["Crossing Number"]), parse_crossing_number)
    else:
        raise NotImplementedError(f"Not implemented target invariant: {args.target_invariant}")
    
    if args.regression_or_classification == "classification":
        label_set_list = sorted(list(set([label[0] for label in y])))
        label_to_index = {label: idx for idx, label in enumerate(label_set_list)}
        y = [[label_to_index[label[0]]] for label in y]
        output_dims = len(label_set_list)
    elif args.regression_or_classification == "regression":
        output_dims = len(y[0])
    else:
        raise ValueError(f"Either regression or classification expected, got: {args.regression_or_classification}")

    if args.notation == "pd":
        notations = parse_list_of_features(list(df["PD Notation"]), parse_pd_notation)
    else:
        raise NotImplementedError(f"Not implemented notation: {args.notation}")

    assert args.number_of_period_in_circular > 0 and args.number_of_period_in_complex_circular > 0, "Number of period in circular and complex circular node embedding should be positive."
    
    if args.data_type == "hyper_graph":
        dataset, max_num_of_nodes = create_hypergraph_dataset_from_pd(notations, y, node_feature_type=args.node_feature_type, embedding_used=args.embedding_used, use_uniform_edge_features=args.uniform_edge_features, classification_or_regression=args.regression_or_classification, number_of_period_in_circular=args.number_of_period_in_circular, number_of_period_in_complex_circular=args.number_of_period_in_complex_circular)
        input_dims = dataset[0].x.shape[1]
    
    random.shuffle(dataset)
    number_of_samples = len(dataset)
    train_size = int(args.train_split_ratio * number_of_samples)
    val_size = int(args.validation_split_ratio * number_of_samples)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    if args.data_loader == "graph":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise NotImplementedError(f"Not implemented data loader: {args.data_loader}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HyperGNN(input_dims=input_dims, hidden_dims=args.hidden_dims, output_dims=output_dims, use_attention=args.use_attention_in_hypergraph, number_of_attention_heads=args.number_of_attention_heads_in_hypergraph, type_of_attention=args.type_of_attention_in_hypergraph, embedding_used=args.embedding_used, max_num_of_nodes=max_num_of_nodes, number_of_layers=args.num_of_layers_in_hypergraph).to(device)
    
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    else:
        raise NotImplementedError(f"Not implemented optimizer: {args.optimizer}")
    
    if args.regression_or_classification == "regression":
        assert args.criterion == 'mse', f"Criterion {args.criterion} is not compatible with regression task."
    elif args.regression_or_classification == "classification":
        assert args.criterion =='cross_entropy', f"Criterion {args.criterion} is not compatible with classification task."
    
    if args.criterion == 'mse':
        criterion = MSELoss()
    elif args.criterion == 'cross_entropy':
        criterion = CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Not implemented criterion: {args.criterion}")

    training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_save_path=args.model_save_path,
        number_of_epochs=args.number_of_epochs,
        early_stopping_patience=args.early_stopping_patience,
        wandb_enabled=args.wandb
    )

    best_model_state = torch.load(args.model_save_path)
    model.load_state_dict(best_model_state)

    model.eval()

    test_predictions, test_targets = create_test_predictions_and_targets(model, test_loader, device)
    test_loss = compute_test_loss(test_predictions, test_targets, criterion)
    if args.regression_or_classification == "regression":
        test_accuracy = compute_test_accuracy_from_mape(test_predictions, test_targets)
        test_r2_score = compute_test_r2_score(test_predictions, test_targets)
        print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.5f} | Test R2 Score: {test_r2_score:.5f}")
        plot_predictions_vs_targets(test_predictions, test_targets, args.figure_save_path, wandb_enabled=args.wandb)

        if args.wandb:
            wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy, "Test R2 Score": test_r2_score})
    elif args.regression_or_classification == "classification":
        predicted_classes = torch.argmax(test_predictions, dim=1)
        test_accuracy = compute_test_accuracy_for_classification(predicted_classes, test_targets)
        test_f1_score = f1_score(predicted_classes, test_targets, average='weighted')
        plot_confusion_matrix(predicted_classes, test_targets, args.figure_save_path, wandb_enabled=args.wandb)
        print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.5f} | Test F1 Score: {test_f1_score:.5f}")
        if args.wandb:
            wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy, "Test F1 Score": test_f1_score})

if __name__ == "__main__":
    main()