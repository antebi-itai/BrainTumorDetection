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
    tumor_name2type   - dictionary mapping tumor's scientific name to type of tumor (integer).
    """
    tumor_type2name = {0: "no", 1: "yes"}
    tumor_name2type = {"no": 0, "yes": 1}

    def __init__(self, data_dir, input_size=(256, 256), reshape_input=True):
        self.data_dir = data_dir
        self.tumor_dirs = os.listdir(self.data_dir)
        self.input_size = input_size
        self.reshape_input = reshape_input
        assert all([tumor_dir.endswith("_tumor") for tumor_dir in self.tumor_dirs])

        # define transforms
        input_transforms = [transforms.ToTensor()]
        if self.reshape_input: input_transforms.insert(0, transforms.Resize(self.input_size))
        self._input_transforms = transforms.Compose(input_transforms)
        self._vis_transforms = transforms.Compose([
            Clip(min=0, max=1),
            transforms.ToPILImage()
        ])

        # parse data from directory to convenient data structures
        self.tumor_image_paths = []
        self.tumor_types = []
        for tumor_dir in self.tumor_dirs:
            tumor_name = tumor_dir.replace("_tumor", "")
            if tumor_name != "no": tumor_name = "yes"
            tumor_type = self.tumor_name2type[tumor_name]
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


class OccludedImageGenerator(Dataset):
    """

    """
    def __init__(self, ref_image_path, occlusion_size=10, stride=1, input_size=(256, 256), reshape_input=True):
        self.ref_image_path = ref_image_path
        self.input_size = input_size
        self.reshape_input = reshape_input

        input_transforms = [transforms.ToTensor()]
        if self.reshape_input: input_transforms.insert(0, transforms.Resize(self.input_size))

        with Image.open(ref_image_path) as tumor_image:
            self.ref_image = input_transforms(tumor_image)

        self._form_occlusions()

    def _form_occlusions(self):
        with Image.open(self.ref_image) as tumor_image:
            tumor_image_input = self._input_transforms(tumor_image)

        # TODO: Sequentially form occlusions (according to stride and size) and save to images and pos correspondingly
        self.images = None
        self.pos = None

    def get_ref_image(self):
        return self.ref_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.pos[idx], self.images[idx]
