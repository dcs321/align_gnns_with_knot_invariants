
import torch
import wandb
import copy

def training_loop(model, train_loader, val_loader, optimizer, criterion, device, data_type, model_save_path, number_of_epochs, early_stopping_patience, wandb_enabled=False):
    best_val_loss = float('inf')
    steps_without_improvement = 0
    best_model = None
    for epoch in range(number_of_epochs):
        model.train()
        train_loss = 0
        for (batch) in train_loader:
            optimizer.zero_grad()
            if data_type == "hyper_graph":
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch.y)
            elif data_type == "pd_notation":
                batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                batch_y = batch_y.squeeze(dim=-1)
                output = model(batch_x)
                loss = criterion(output, batch_y.to(device))
            loss.backward()
            optimizer.step()
            if data_type == "hyper_graph":
                train_loss += loss.item() * batch.num_graphs
            elif data_type == "pd_notation":
                train_loss += loss.item() * batch_x.shape[0]

        average_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if data_type == "hyper_graph":
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch.y)
                    val_loss += loss.item() * batch.num_graphs
                elif data_type == "pd_notation":
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    batch_y = batch_y.squeeze(dim=-1)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item() * batch_x.shape[0]

        average_val_loss = val_loss / len(val_loader.dataset)
        if wandb_enabled:
            wandb.log({
                'Epoch': epoch + 1,
                'Train Loss': average_train_loss,
                'Validation Loss': average_val_loss
            })
        print(f"Epoch {epoch+1} | Train Loss: {average_train_loss:.5f} | Val Loss: {average_val_loss:.5f}")

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            steps_without_improvement = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            steps_without_improvement += 1
            print(f"No improvement in validation loss for {steps_without_improvement} epochs.")

        if steps_without_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    torch.save(best_model, model_save_path)
    if wandb_enabled:
        wandb.save(model_save_path)
    return best_model