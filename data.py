import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


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
    """

    def __init__(self, data_dir, input_size=(256, 256)):
        self.data_dir = data_dir
        self.tumor_dirs = os.listdir(self.data_dir)
        self.input_size = input_size
        assert all([tumor_dir.endswith("_tumor") for tumor_dir in self.tumor_dirs])

        # define transforms
        self._input_transforms = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self._vis_transforms = transforms.Compose([
            Clip(min=0, max=1),
            transforms.ToPILImage()
        ])

        # parse data from directory to convenient data structures
        self.tumor_image_paths = []
        self.tumor_types = []
        self.tumor_type2name = {}
        for tumor_type, tumor_dir in enumerate(self.tumor_dirs):
            # add tumor name to type2name dict
            tumor_name = tumor_dir.replace("_tumor", "")
            self.tumor_type2name[tumor_type] = tumor_name
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
