import argparse
import pandas as pd
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import wandb

from data_parsers import parse_volume, parse_pd_notation, parse_list_of_features
from datasets import create_hypergraph_dataset_from_pd
from models import HyperGNN
from training import training_loop
from evaluation import create_test_predictions_and_targets, compute_test_loss, compute_test_accuracy_from_mape, plot_predictions_vs_targets

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
    parser.add_argument('--number_of_epochs', type=int, default=200, required=False, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=15, required=False, help='Early stopping patience')
    parser.add_argument('--model_save_path', type=str,default="models/model.pth", required=False, help='Model save path')
    parser.add_argument('--figure_save_path', type=str, default="figures/plot.png", required=False, help='Figure save path')
    parser.add_argument('--hidden_dims', type=int, default=64, required=False, help='Hidden dimensions')
    parser.add_argument('--device', type=str, default='cpu', required=False, help='Device (cpu or cuda)')
    parser.add_argument('--criterion', type=str, default='mse', required=False, help='Criterion')
    parser.add_argument('--optimizer', type=str, default='adam', required=False, help='Optimizer')
    parser.add_argument('--random_seed', type=int, default=42, required=False, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')


    args = parser.parse_args()

    if args.wandb:
        wandb.init(config=vars(args))
 
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    df = pd.read_csv(args.data_path)

    if args.target_invariant == "volume":
        y = parse_list_of_features(list(df["Volume"]), parse_volume)
    else:
        raise NotImplementedError(f"Not implemented target invariant: {args.target_invariant}")
    output_dims = len(y[0])

    if args.notation == "pd":
        notations = parse_list_of_features(list(df["PD Notation"]), parse_pd_notation)
    else:
        raise NotImplementedError(f"Not implemented notation: {args.notation}")
    
    if args.data_type == "hyper_graph":
        dataset = create_hypergraph_dataset_from_pd(notations, y, node_feature_type=args.node_feature_type)
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
    model = HyperGNN(input_dims=input_dims, hidden_dims=args.hidden_dims, output_dims=output_dims).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()

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
    test_accuracy = compute_test_accuracy_from_mape(test_predictions, test_targets)
    print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.5f}")
    plot_predictions_vs_targets(test_predictions, test_targets, args.figure_save_path, wandb_enabled=args.wandb)

    if args.wandb:
        wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})

if __name__ == "__main__":
    main()