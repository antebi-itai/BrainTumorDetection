import torch
from torch.autograd import backward


def train_epoch(model, criterion, accuracy, optimizer, loader, device):
    for images, tumor_types in loader:
        images, tumor_types = images.to(device=device), tumor_types.to(device=device)
        # Run the model on the input batch
        pred_tumors_scores = model(images)

        # special case - binary classification. must change gt types
        if pred_tumors_scores.size(-1) == 2:
            no_tumor_type = loader.dataset.tumor_name2type["no"]
            tumor_types = torch.where(tumor_types == no_tumor_type, torch.zeros_like(tumor_types),
                                      torch.ones_like(tumor_types))

        # Calculate the loss (and acc) for this batch
        loss = criterion(pred_tumors_scores, tumor_types)
        acc = accuracy(pred_tumors_scores, tumor_types)
        # Calculate the gradients of all parameter w.r.t. the loss
        optimizer.zero_grad()
        backward(loss)
        # Update the weights (learn!)
        optimizer.step()

    return loss, acc


def train(model, criterion, accuracy, optimizer, loader, epochs, device):
    loss_metric = []
    acc_metric = []

    for i in range(epochs):
        loss, acc = train_epoch(model=model, criterion=criterion, accuracy=accuracy,
                                optimizer=optimizer, loader=loader, device=device)

        loss_metric.append(loss.item())
        acc_metric.append(acc.item())

    return loss_metric, acc_metric
