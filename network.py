from torch.nn import Module, Linear
import torchvision
from train import MODELS_DIR
import torch
import os


def get_model_and_optim(model_name, pretrained=True, num_class=2, lr=1e-4, load_best_model=True, device="cuda"):
    # define model
    if model_name == 'vgg19':
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        model.classifier[-1] = Linear(in_features=4096, out_features=num_class, bias=True)
    else:
        raise NotImplementedError
    setattr(model, "name", model_name)
    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # load best model & optimizer state
    if load_best_model:
        best_model_path = os.path.join(MODELS_DIR, model.name + ".pth")
        if os.path.exists(best_model_path):
            print("loading best model from: {0}".format(best_model_path))
            states_dict = torch.load(best_model_path)
            model.load_state_dict(states_dict['model_state_dict'])
            optimizer.load_state_dict(states_dict['optimizer_state_dict'])

    return model, optimizer
