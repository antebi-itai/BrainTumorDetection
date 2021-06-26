import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import util
import random


class Clip:
    """Clips the values of an tensor into the requested range."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        return torch.clip(tensor, self.min, self.max)


class DataGenerator(Dataset):
    """
    Generates supervised tumor data for training - (image, tumor_type)

    tumor_image_paths - list of paths of all images in the dataset
    tumor_types       - list of types of tumors corresponding to tumor_image_paths.
                        i.e. tumor_types[idx] is the type of tumor in image tumor_image_paths[idx]
    tumor_type2name   - dictionary mapping type of tumor (integer) to tumor's scientific name.
    tumor_name2type   - dictionary mapping tumor's scientific name to type of tumor (integer).
    """

    def __init__(self, data_dir, input_size=(256, 256), reshape_input=True):
        self.data_dir = data_dir
        self.tumor_dirs = os.listdir(self.data_dir)
        self.input_size = input_size
        self.reshape_input = reshape_input
        assert all([tumor_dir.endswith("_tumor") for tumor_dir in self.tumor_dirs])

        # define transforms
        input_transforms = [transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)]
        if self.reshape_input: input_transforms.insert(0, transforms.Resize(self.input_size))
        self._input_transforms = transforms.Compose(input_transforms)
        self._vis_transforms = transforms.Compose([
            Clip(min=0, max=1),
            transforms.ToPILImage()
        ])

        # parse data from directory to convenient data structures
        self.tumor_image_paths = []
        self.tumor_types = []
        self.tumor_type2name = {}
        self.tumor_name2type = {}
        for tumor_type, tumor_dir in enumerate(self.tumor_dirs):
            # add tumor name to type2name dict
            tumor_name = tumor_dir.replace("_tumor", "")
            self.tumor_type2name[tumor_type] = tumor_name
            self.tumor_name2type[tumor_name] = tumor_type
            # add tumor image paths and types to lists
            tumor_image_paths = [os.path.join(self.data_dir, tumor_dir, file_name)
                                 for file_name in os.listdir(os.path.join(self.data_dir, tumor_dir))]
            self.tumor_image_paths += tumor_image_paths
            self.tumor_types += [tumor_type] * len(tumor_image_paths)
        assert len(self.tumor_image_paths) == len(self.tumor_types)

    def __len__(self):
        return len(self.tumor_image_paths)

    def __getitem__(self, idx):
        tumor_label = self.tumor_types[idx]
        tumor_image_path = self.tumor_image_paths[idx]
        with Image.open(tumor_image_path) as tumor_image:
            tumor_image_input = self._input_transforms(tumor_image)
        return tumor_image_input, tumor_label


class RandomOccludedDataGenerator(DataGenerator):
    def __init__(self, data_dir, occlusion_size=(50, 50), input_size=(256, 256), reshape_input=True):
        super().__init__(self, data_dir, input_size, reshape_input)
        self.occlusion_size = occlusion_size

    def __getitem__(self, idx):
        tumor_image_input, tumor_label = super().__getitem__(idx)
        rand_occlusion = random.randint(0, tumor_image_input.shape[0] * tumor_image_input.shape[1] - 1)

        return util.occlude_image(tumor_image_input, rand_occlusion, self.occlusion_size)


class OccludedImageGenerator(Dataset):
    def __init__(self, ref_image_path, occlusion_size=(50, 50), reshape_input=True, reshape_input_size=(256, 256)):
        self.ref_image_path = ref_image_path
        self.reshape_input = reshape_input
        self.occlusion_size = occlusion_size

        input_transforms = [transforms.ToTensor(), transforms.Grayscale(num_output_channels=3)]
        if self.reshape_input: input_transforms.insert(0, transforms.Resize(reshape_input_size))
        input_filter = transforms.Compose(input_transforms)

        with Image.open(ref_image_path) as tumor_image:
            self.ref_image = input_filter(tumor_image)

    def get_image(self):
        return self.ref_image

    def __len__(self):
        return self.ref_image.shape[1] * self.ref_image.shape[2]

    def __getitem__(self, idx):
        image = torch.clone(self.ref_image)
        util.occlude_image(image, idx, occlusion_size=self.occlusion_size)

        return idx, image
