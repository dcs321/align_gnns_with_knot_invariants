import torch
import matplotlib.pyplot as plt
import wandb

def create_test_predictions_and_targets(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            test_targets.append(batch.y.cpu())
            test_predictions.append(output.cpu())
    test_targets = torch.cat(test_targets, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)
    return test_predictions, test_targets

def compute_test_loss(test_predictions, test_targets, criterion):
    test_loss = criterion(test_predictions, test_targets)
    return test_loss.item()

def compute_test_accuracy_from_mape(test_predictions, test_targets):
    relative_errors = torch.abs((test_targets - test_predictions) / (test_targets + 1e-10))
    mape = torch.mean(relative_errors).item()
    accuracy = 1 - mape
    return accuracy


def  plot_predictions_vs_targets(test_predictions, test_targets,figure_save_path, wandb_enabled=False):
    min_value = min(test_targets.min(), test_predictions.min()).item()
    max_value = max(test_targets.max(), test_predictions.max()).item()
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(test_targets, test_predictions)
    plt.plot([min_value, max_value], [min_value, max_value], 'r--')
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Targets')
    plt.grid()
    plt.savefig(figure_save_path)
    plt.close(fig)
    
    if wandb_enabled:
        wandb.log({"regression_plot": wandb.Image(figure_save_path)})