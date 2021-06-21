import torch
from torchvision import models

##########################################################
# Feature Extractor
##########################################################

class FeatureExtractor:
  """This class facilitates extracting features from a pretrained network """

  def __init__(self, model, layers, device='cuda'):
    """
    Args:
      device (string): Device to host model on. Default host on cuda.
      model (model):
    """

    # Save arguments
    self.model = model
    self.layers = layers

    # Make room for layers output
    self.layers_output = {}
    for layer in layers:
      self.layers_output[layer] = []

    # initiate a list to hold the hooks handlers.
    self.hook_handlers = []

  def _get_hook(self, layer):
    """
    Defines the hook for extracting the features of `layer`
    Returns:            
      The method to hook. 
    """
    def _get_layer_output(model, input, output):
      self.layers_output[layer].append(output)
    return _get_layer_output

  def plug(self):
    """
    Registers all the hooks to perform extraction.
    Args:
      kwargs (dict(string, List(int))): dictionary with all keys in KEY_LIST.
      Each key's value is a list of all the layers to extract for this key.
    """
    for layer in self.layers:
      hook = self._get_hook(layer)
      handle = self.model.features[layer].register_forward_hook(hook)
      self.hook_handlers.append(handle)

  def unplug(self):
    """
    Unregisters all the hooks after performing an extraction.
    """
    # BEGIN SOLUTION
    for hook_handle in self.hook_handlers:
      hook_handle.remove()
    # END SOLUTION

  def flush_features(self):
    """
    Returns the features
    """
    # Copy and clean the slate
    layers_output = dict(self.layers_output)

    for layer in self.layers:
      self.layers_output[layer] = []

    return layers_output



