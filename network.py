from torch.nn import Module, Linear
import torchvision


class VGGNet(Module):

	def __init__(self, vgg_version="19", pretrained=True, num_classes=2):
		super(VGGNet, self).__init__()
		self.model = getattr(torchvision.models, "vgg" + vgg_version)(pretrained=pretrained)
		self.model.classifier[-1] = Linear(in_features=4096, out_features=num_classes, bias=True)

	def forward(self, x):
		return self.model(x)
