import torch
from train import train, train_loop
from loss import calc_accuracy
from feature_extractor import FeatureExtractor
from data import OccludedImageGenerator
from network import get_model_and_optim, load_best_state
import wandb
wandb.login()
from tqdm import tqdm
import cv2
import os
import numpy as np


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        for key, value in config.items():
            setattr(self, key, value)

        # train
        self.train_dataset = self.data_train_class(data_dir=self.data_train_path, mri_type=self.mri_type)
        train_weights = self.train_dataset.make_weights_for_balanced_classes()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.train_batch_size, sampler=train_sampler)
        # test
        self.test_dataset = self.data_test_class(data_dir=self.data_test_path, mri_type=self.mri_type)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.train_batch_size, shuffle=True)

        self.criterion = torch.nn.functional.cross_entropy
        self.calc_accuracy = calc_accuracy
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion, calc_accuracy=self.calc_accuracy,
              optimizer=self.optimizer, train_loader=self.train_loader, test_loader=self.test_loader,
              epochs=self.epochs, device=self.device)
        load_best_state(self.model, self.optimizer)

    """
    TODO: Doc
    """
    def eval_model(self):
        accuracies = []
        weights = []
        for (test_images, (test_tumor_segmentations, test_tumor_types)) in self.test_loader:
            # test accuracy of batch
            accuracy = train_loop(model=self.model, criterion=self.criterion, calc_accuracy=self.calc_accuracy,
                                  optimizer=self.optimizer, device=self.device,
                                  images=test_images, tumor_types=test_tumor_types, mode="Test")
            accuracies.append(accuracy)
            weights.append(test_images.size(0))
        test_accuracy = sum([accuracy * weight for accuracy, weight in zip(accuracies, weights)]) / sum(weights)
        return test_accuracy

    """
    TODO: Doc
    """
    def generate_heatmap(self, image_path):
        """
        Generate heat maps from image, based on experiment's model and heat layers

        :param image_path: path to the image from which to create the heat maps
        :return: heatmaps: dictionary containing descriptive keys (L`layer`C`channel) and corresponding heatmap.
                           ordered in the same manner as the heat layers in the configuration
        """
        occluded_loader = torch.utils.data.DataLoader(OccludedImageGenerator(image_path, occlusion_size=self.occlusion_size),
                                                      batch_size=self.heatmap_batch_size, shuffle=False)

        # Start extracting features
        fe = FeatureExtractor(model=self.model, device=self.device)

        # Hook heat layers
        for heat_layer in self.heat_layers:
            fe.plug_layer(heat_layer)

        # Forward pass over original image
        self.model.eval()
        original_image = occluded_loader.dataset.get_image().to(self.device).unsqueeze(0)
        height, width = original_image.shape[2], original_image.shape[3]

        self.model(original_image)
        feature_layers = fe.flush_layers()

        # TODO: REMOVE
        from matplotlib import pyplot as plt

        for i in range(256, 286):
            fe.plug_activation(['features', 0], 29)
            x = occluded_loader.dataset[i][1].to(self.device).unsqueeze(0)
            plt.figure(); plt.imshow(occluded_loader.dataset[i][1][0,:,:], cmap="gray"); plt.show()
            y = self.model(x)
            activations = fe.flush_activations()
            print("Activation #{}:{}".format(i, activations["L['features', 0]C29"][0].cpu()))
        # TODO: UP TO HERE

        # Hook the channel with max value
        for layer_pos, (_, feature_layer) in zip(self.heat_layers, feature_layers.items()):
            if len(feature_layer[0].shape) == 4:
                # Set the most 'activated channel' as reference
                channel = torch.argmax(torch.sum(feature_layer[0], dim=(2, 3))).item()  # Sum over spatial dimensions
                fe.plug_activation(layer_pos, channel)
            else:
                fe.plug_layer(layer_pos)

        # Forward pass over all occlusions of an image
        print("Creating heatmap...", flush=True)
        with torch.no_grad():
            for idx, images in tqdm(occluded_loader):
                occluded_images = images.to(device=self.device)
                self.model(occluded_images)

                wandb.log({"memory/usage": torch.cuda.memory_allocated()/(1024**2)})

        activations = fe.flush_activations()
        layers = fe.flush_layers()

        heatmaps = {}
        overlay_heatmaps = {}
        # Generate Convolutional heatmaps
        for channel, heatmap_array in activations.items():
            gray_heatmap = torch.cat(heatmap_array).reshape((height, width)).cpu().numpy()
            gray_heatmap = np.uint8(cv2.normalize(gray_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[channel] = gray_heatmap
            np_original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            np_original_image = np.uint8(cv2.normalize(np_original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            overlay_heatmaps[channel] = 0.5 * colorful_heatmap + 0.5 * np_original_image
            
        # Generate Linear layers heatmaps
        for layer, heatmap_array in layers.items():
            heatmap = torch.cat(heatmap_array).reshape((height, width, -1))
            correlated_heatmap = heatmap[:, :, 1] - heatmap[:, :, 0]
            gray_heatmap = correlated_heatmap.cpu().numpy()
            gray_heatmap = np.uint8(cv2.normalize(gray_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[layer] = gray_heatmap
            np_original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            np_original_image = np.uint8(cv2.normalize(np_original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            overlay_heatmaps[layer] = 0.5 * colorful_heatmap + 0.5 * np_original_image

        # Log heatmaps
        image_name = os.path.basename(image_path)
        wandb.log({"heatmaps/grayscale {image_name}".format(image_name=image_name):
                       [wandb.Image(gray_heatmap, caption=channel) for channel, gray_heatmap in heatmaps.items()]})
        wandb.log({"heatmaps/overlay {image_name}".format(image_name=image_name):
                       [wandb.Image(overlay_heatmap, caption=channel) for channel, overlay_heatmap in overlay_heatmaps.items()]})

        return heatmaps
