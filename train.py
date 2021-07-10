from data import tumor_type2name
from torch.autograd import backward
import wandb
from tqdm import tqdm
from itertools import cycle
import os
import pickle
import torch

MODELS_DIR = os.path.join(".", "Models")
BEST_MODELS_DICT_PATH = os.path.join(MODELS_DIR, "best_models_dict.pkl")


def get_best_models_dict():
    if os.path.exists(BEST_MODELS_DICT_PATH):
        with open(BEST_MODELS_DICT_PATH, "rb") as f:
            best_models_dict = pickle.load(f)
    else:
        best_models_dict = {}
    return best_models_dict


def update_best_models(model, optimizer, model_acc, best_models_dict):
    print("Best model found!!", "Model: {0}, Accuracy: {1}".format(model.name, model_acc))
    # update dict
    best_models_dict[model.name] = model_acc
    with open(BEST_MODELS_DICT_PATH, "wb") as f:
        pickle.dump(best_models_dict, f)
    # update model
    best_model_path = os.path.join(MODELS_DIR, model.name + ".pth")
    print("saving best model to: {0}".format(best_model_path))
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, best_model_path)


def train(model, criterion, calc_accuracy, optimizer, train_loader, test_loader, epochs, device):
    for epoch in tqdm(range(epochs)):
        for (train_images, (train_tumor_segmentations, train_tumor_types)), \
            (test_images, (test_tumor_segmentations, test_tumor_types)) in zip(train_loader, cycle(test_loader)):

            # train
            train_model_acc = train_loop(model=model, criterion=criterion, calc_accuracy=calc_accuracy,
                                         optimizer=optimizer,
                                         device=device, images=train_images, tumor_types=train_tumor_types,
                                         mode="Train")
            # test
            test_model_acc = train_loop(model=model, criterion=criterion, calc_accuracy=calc_accuracy,
                                        optimizer=optimizer,
                                        device=device, images=test_images, tumor_types=test_tumor_types, mode="Test")
            # update best model if necessary
            best_models_dict = get_best_models_dict()
            if model.name not in best_models_dict or test_model_acc > best_models_dict[model.name]:
                update_best_models(model=model, optimizer=optimizer, model_acc=test_model_acc,
                                   best_models_dict=best_models_dict)


def train_loop(model, criterion, calc_accuracy, optimizer, device, images, tumor_types, mode="Train"):
    # Set model mode
    if mode == "Train":
        model.train()
    elif mode == "Test":
        model.eval()
    else:
        raise RuntimeError()

    # Move to device
    images, tumor_types = images.to(device=device), tumor_types.to(device=device)

    # Run the model on the input batch
    pred_tumors_scores = model(images)

    min_acc = None
    # Calculate the accuracy for this batch
    for tumor_type, tumor_name in tumor_type2name.items():
        tumor_acc = calc_accuracy(pred_tumors_scores, tumor_types, tumor_type=tumor_type)
        wandb.log({"{mode}/accuracy/{tumor_name}".format(mode=mode, tumor_name=tumor_name): tumor_acc})
        min_acc = tumor_acc if min_acc is None else min(min_acc, tumor_acc)
    accuracy = calc_accuracy(pred_tumors_scores, tumor_types, tumor_type=None)
    wandb.log({"{mode}/accuracy".format(mode=mode): accuracy})

    if mode == "Train":
        # Calculate the loss for this batch
        loss = criterion(pred_tumors_scores, tumor_types)
        wandb.log({"{mode}/loss".format(mode=mode): loss})
        # Update gradients
        optimizer.zero_grad()
        backward(loss)
        optimizer.step()

    return accuracy
