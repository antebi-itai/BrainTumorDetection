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
            classification_size = pred_tumors_scores.size(-1)
            tumor_names = [loader.dataset.tumor_type2name[tumor_type] for tumor_type in range(classification_size)]
            if classification_size == 2:
                no_tumor_type = loader.dataset.tumor_name2type["no"]
                tumor_types = torch.where(tumor_types == no_tumor_type, torch.zeros_like(tumor_types),
                                          torch.ones_like(tumor_types))
                tumor_names = ["no", "yes"]

            # Calculate the loss (and acc) for this batch
            loss = criterion(pred_tumors_scores, tumor_types)
            wandb.log({"Train/loss": loss})
            for tumor_type, tumor_name in enumerate(tumor_names):
                tumor_acc = accuracy(pred_tumors_scores, tumor_types, tumor_type)
                wandb.log({"Train/accuracy/{tumor_name}".format(tumor_name=tumor_name): tumor_acc})
            acc = accuracy(pred_tumors_scores, tumor_types, tumor_type=None)
            wandb.log({"Train/accuracy": acc})

            # Calculate the gradients of all parameter w.r.t. the loss
            optimizer.zero_grad()
            backward(loss)
            # Update the weights (learn!)
            optimizer.step()
