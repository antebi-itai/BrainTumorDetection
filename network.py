from torch.nn import Module, Linear
import torchvision


def get_model(self, model_name='vgg19', pretrained=True, num_class=2):
	if model_name == 'vgg19':
		model = getattr(torchvision.models, model_name)(pretrained=pretrained)
		model.classifier[-1] = Linear(in_features=4096, out_features=num_class, bias=True)
	else:
		raise NotImplementedError

	return model