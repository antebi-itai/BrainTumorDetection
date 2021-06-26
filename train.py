from data import DataGenerator
from torch.autograd import backward
import wandb
from tqdm import tqdm
from itertools import cycle


def train(model, criterion, accuracy, optimizer, train_loader, test_loader, epochs, device):

    for epoch in tqdm(range(epochs)):
        for (images, tumor_types), (test_images, test_tumor_types) in zip(train_loader, cycle(test_loader)):
            train_loop(model=model, criterion=criterion, accuracy=accuracy, optimizer=optimizer, device=device,
                       images=images, tumor_types=tumor_types, mode="Train")
            train_loop(model=model, criterion=criterion, accuracy=accuracy, optimizer=optimizer, device=device,
                       images=test_images, tumor_types=test_tumor_types, mode="Test")


def train_loop(model, criterion, accuracy, optimizer, device, images, tumor_types, mode="Train"):
    # Move to device
    images, tumor_types = images.to(device=device), tumor_types.to(device=device)
    # Run the model on the input batch
    pred_tumors_scores = model(images)

    # Calculate the accuracy for this batch
    for tumor_type, tumor_name in DataGenerator.tumor_type2name.items():
        tumor_acc = accuracy(pred_tumors_scores, tumor_types, tumor_type=tumor_type)
        wandb.log({"{mode}/accuracy/{tumor_name}".format(mode=mode, tumor_name=tumor_name): tumor_acc})
    acc = accuracy(pred_tumors_scores, tumor_types, tumor_type=None)
    wandb.log({"{mode}/accuracy".format(mode=mode): acc})

    if mode == "Train":
        # Calculate the loss for this batch
        loss = criterion(pred_tumors_scores, tumor_types)
        wandb.log({"{mode}/loss".format(mode=mode): loss})
        # Update gradients
        optimizer.zero_grad()
        backward(loss)
        optimizer.step()
