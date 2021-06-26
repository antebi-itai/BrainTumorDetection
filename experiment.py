import torch
from train import train
from loss import accuracy
from feature_extractor import FeatureExtractor
from data import DataGenerator, OccludedImageGenerator
from network import get_model


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        for key, value in config.items():
            setattr(self, key, value)

        # training setting
        self.criterion = torch.nn.functional.cross_entropy
        self.accuracy = accuracy
        self.test_dataset = DataGenerator(self.data_test_path)
        self.train_dataset = DataGenerator(self.data_train_path)
        weights = self.train_dataset.make_weights_for_balanced_classes()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=sampler)
        self.model = get_model(self.model_name).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion, accuracy=self.accuracy,
              optimizer=self.optimizer, train_loader=self.train_loader, test_loader=self.test_loader,
              epochs=self.epochs, device=self.device)

    """
    TODO: Doc
    """
    def generate_heatmap(self, image_path):
        occluded_loader = torch.utils.data.DataLoader(OccludedImageGenerator(image_path, occlusion_size=self.occlusion_size),
                                                      batch_size=self.batch_size, shuffle=False)

        # Start extracting features
        fe = FeatureExtractor(model=self.model, device=self.device)

        for heat_layer in self.heat_layers:
          fe.plug_layer(heat_layer)

        # Forward pass over original image & Plug the channel with max value
        original_image = occluded_loader.dataset.get_image().to(self.device).unsqueeze(0)
        self.model(original_image)

        feature_layers = fe.flush_layers()

        for layer, feature_layer in feature_layers.items():
          # Set the most 'activated channel' as reference
          channel = torch.argmax(torch.sum(feature_layer, dim=(2,3))).item()  # Sum over spatial dimensions
          fe.plug_activation(layer, channel)

        # Gather heatmap - forward pass over all occlusioned images & generate corresponding heatmaps
        heatmap = []
        for idx, images in occluded_loader:
          occluded_images = images.to(device=self.device)
          # Run the model on batched occluded images
          self.model(occluded_images)

        features = fe.flush_activations()
        return torch.cat(features).reshape(original_image.shape)
