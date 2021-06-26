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

            # Calculate the loss (and acc) for this batch
            loss = criterion(pred_tumors_scores, tumor_types)
            wandb.log({"Train/loss": loss})
            for tumor_type, tumor_name in loader.dataset.tumor_type2name.items():
                tumor_acc = accuracy(pred_tumors_scores, tumor_types, tumor_type=tumor_type)
                wandb.log({"Train/accuracy/{tumor_name}".format(tumor_name=tumor_name): tumor_acc})
            acc = accuracy(pred_tumors_scores, tumor_types, tumor_type=None)
            wandb.log({"Train/accuracy": acc})

            # Calculate the gradients of all parameter w.r.t. the loss
            optimizer.zero_grad()
            backward(loss)
            # Update the weights (learn!)
            optimizer.step()
