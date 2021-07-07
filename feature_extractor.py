import torch
from torchvision import models


##########################################################
# Feature Extractor
##########################################################

class FeatureExtractor:
    """This class facilitates extracting features from a pretrained network """

    def __init__(self, model, device='cuda'):
        """
        Args:
          device (string): Device to host model on. Default host on cuda.
          model (model):
        """

        # Save arguments
        self.model = model

        # Make room for layers output
        self.layers_output = {}
        self.activations_output = {}

        # initiate a list to hold the hooks handlers.
        self.layer_hook_handlers = []
        self.activation_hook_handlers = []

    def _traverse_layers(self, layer_pos):
        runner = self.model._modules
        for key in layer_pos:
            runner = runner[key]

        return runner

    def _init_layer_hook(self, layer):
        """
        Defines the hook for extracting the features of `layer`
        Returns:
          The method to hook.
        """
        self.layers_output[layer] = []

        def _get_layer_output(model, input, output):
            self.layers_output[layer].append(output)

        return _get_layer_output

    def plug_layer(self, layer_pos):
        """

        :param layer_pos: the position of the layer within the model, given as array
        :return:
        """
        # Traverse through layers
        layer = self._traverse_layers(layer_pos)
        hook = self._init_layer_hook(str(layer_pos))
        handle = layer.register_forward_hook(hook)
        self.layer_hook_handlers.append(handle)

    def _init_activation_hook(self, layer_pos, channel):
        key = 'L' + str(layer_pos) + 'C' + str(channel)
        self.activations_output[key] = []

        def _get_channel_activation(model, input, output):
            channel_activation = torch.sum(output[:, channel, :, :], dim=(1, 2))
            self.activations_output[key].append(channel_activation)

        return _get_channel_activation

    def plug_activation(self, layer_pos, channel):
        layer = self._traverse_layers(layer_pos)
        hook = self._init_activation_hook(layer_pos, channel)
        handle = layer.register_forward_hook(hook)
        self.activation_hook_handlers.append(handle)

    def flush_layers(self):
        """
        Returns the features
        """
        # Copy and clean the slate
        layers_output = dict(self.layers_output)
        self.layers_output = {}

        for handle in self.layer_hook_handlers:
            handle.remove()

        return layers_output

    def flush_activations(self):
        """
        Returns the features
        """
        # Copy and clean the slate
        activations_output = dict(self.activations_output)
        self.activations_output = {}

        for handle in self.activation_hook_handlers:
            handle.remove()

        return activations_output
