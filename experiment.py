import torch
from train import train, train_loop
from loss import accuracy
from feature_extractor import FeatureExtractor
from data import DataGenerator, OccludedImageGenerator
from network import get_model_and_optim, load_best_state
from util import normalize_numpy
import wandb
wandb.login()
from tqdm import tqdm
import cv2


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        for key, value in config.items():
            setattr(self, key, value)

        # Settings
        self.criterion = torch.nn.functional.cross_entropy
        self.accuracy = accuracy
        self.test_dataset = DataGenerator(self.data_test_path)
        self.train_dataset = DataGenerator(self.data_train_path)
        weights = self.train_dataset.make_weights_for_balanced_classes()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size, sampler=sampler)
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion, accuracy=self.accuracy,
              optimizer=self.optimizer, train_loader=self.train_loader, test_loader=self.test_loader,
              epochs=self.epochs, device=self.device)
        load_best_state(self.model, self.optimizer)

    """
    TODO: Doc
    """
    def eval_model(self):
        accuracies = []
        for test_images, test_tumor_types in self.test_loader:
            accuracy = train_loop(model=self.model, criterion=self.criterion, accuracy=self.accuracy,
                                  optimizer=self.optimizer, device=self.device,
                                  images=test_images, tumor_types=test_tumor_types, mode="Test")
            accuracies.append(accuracy)
        return (sum(accuracies) / len(accuracies)).item()

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
        original_image = occluded_loader.dataset.get_image().to(self.device).unsqueeze(0)
        height, width = original_image.shape[2], original_image.shape[3]

        self.model(original_image)
        feature_layers = fe.flush_layers()

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
            # Run the model on batched occluded images
            self.model(occluded_images)
            wandb.log({"memory/usage": torch.cuda.memory_allocated()/(1024**2)})
        features = fe.flush_activations()
        layers = fe.flush_layers()

        # Generate convolutional heatmaps
        heatmaps = {}
        overlay_heatmaps = {}
        for channel, heatmap in features.items():
            gray_heatmap = normalize_numpy(torch.cat(heatmap).reshape((height, width)).cpu().numpy())
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[channel] = gray_heatmap
            overlay_heatmaps[channel] = 0.5 * colorful_heatmap + \
                                        0.5 * normalize_numpy(original_image.squeeze().permute(1, 2, 0).cpu().numpy())

        # Generate linear layers heatmaps
        softmax = torch.nn.Softmax(dim=2)
        for layer, heatmap in layers.items():
            gray_heatmap = normalize_numpy(softmax(torch.cat(heatmap).reshape((height, width, -1)))[:, :, 0].cpu().numpy())
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[layer] = gray_heatmap
            overlay_heatmaps[layer] = 0.5 * colorful_heatmap + \
                                        0.5 * normalize_numpy(original_image.squeeze().permute(1, 2, 0).cpu().numpy())

        # Log heatmaps
        wandb.log({"heatmaps/grayscale": [wandb.Image(gray_heatmap, caption=channel) for channel, gray_heatmap in heatmaps.items()]})
        wandb.log({"heatmaps/overlay": [wandb.Image(overlay_heatmap, caption=channel) for channel, overlay_heatmap in overlay_heatmaps.items()]})

        return heatmaps
