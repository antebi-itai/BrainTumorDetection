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
        self.model = get_model(self.model_name).to(self.device)
        self.criterion = torch.nn.functional.cross_entropy
        self.accuracy = accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loader = torch.utils.data.DataLoader(DataGenerator(self.data_train_path),
                                                  batch_size=self.batch_size, shuffle=self.shuffle_data)

    """
    TODO: Doc
    """
    def train_model(self):
        return train(model=self.model, criterion=self.criterion, accuracy=self.accuracy,
                     optimizer=self.optimizer, loader=self.loader, epochs=self.epochs, device=self.device)

    """
    TODO: Doc
    """
    def generate_heatmap(self, image_path):
        occluded_loader = torch.utils.data.DataLoader(OccludedImageGenerator(image_path,
                                                                             occlusion_size=self.occlusion_size),
                                                      batch_size=self.batch_size, shuffle=self.shuffle_data)

        # Start extracting features
        fe = FeatureExtractor(model=self.model, layers=[1], device=self.device)
        fe.plug()

        # Forward pass over reference image & Flag (per layer) the `ref_channel` as the channel with max value
        ref_channel = {}
        import pdb; pdb.set_trace() #TODO: Problem here is it expects image with 3 channels, however our's greyscale
        self.model(occluded_loader.dataset.get_image().unsqueeze(0).unsqueeze(0))
        features = fe.flush_features()

        for layer, feature_layer in features:
            # TODO: change 2,3 to right dims
            ref_channel[layer] = torch.amax(torch.sum(feature_layer[layer][0], dim=[2, 3]))  # Sum over spatial dimensions

        # Forward pass over all images with occlusions & Generate corresponding heatmaps
        pts = self._inference(occluded_loader)
        features = fe.flush_features()

        # features should be a dict of layers. Each such element is a list of the layer's features when a
        # forward pass on the (x,y) occluded image was done.
        # The order should correspond to the list of occluded image centers given by self._inference (pts)
        for feature_layer_batch in features:
            for (x, y), feature_layer in zip(pts, feature_layer_batch):
                # perform sum over spatial dimensions on corresponding `max-channel`
                channel = ref_channel[layer]
                # TODO: change 2,3 to right dims
                # TODO: Batch indices and layers indices would be tricky. Need to figure it out and fix it.
                heatmap[layer][x][y] = torch.sum(feature_layer[channel], dim=[2,3])

        # TODO: Draw the heatmaps per each layer

        fe.unplug()
