import torch
from torch.autograd import backward
import wandb
from tqdm import tqdm


def train(model, criterion, accuracy, optimizer, loader, epochs, device):

    for epoch in tqdm(range(epochs)):
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
            # log
            wandb.log({"Train/loss" : loss})
            wandb.log({"Train/accuracy": acc})