import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import util
import random
import SimpleITK as sitk
import numpy as np


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
        input_transforms = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
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

    def make_weights_for_balanced_classes(self):
        nclasses = len(self.tumor_type2name)

        count = [0] * nclasses
        for tumor_type in self.tumor_types:
            count[tumor_type] += 1

        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        weights = [0] * len(self.tumor_types)
        for idx, tumor_type in enumerate(self.tumor_types):
            weights[idx] = weight_per_class[tumor_type]
        return torch.DoubleTensor(weights)


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

        input_transforms = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
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


def get_gt_mha_file_path(brain_dir):
    for file_name in os.listdir(brain_dir):
        if SegmentationGenerator.GT_PATTERN in file_name:
            return os.path.join(brain_dir, file_name)


def np_from_mha_path(mha_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(mha_path))


def get_most_activated_slice(np_arr):
    assert len(np_arr.shape) == 3
    peak_slice = np_arr.sum(axis=(1, 2)).argmax()
    return peak_slice


class SegmentationGenerator(Dataset):

    GT_PATTERN = ".OT."
    MRI_TYPES = ["Flair", "T1", "T1c", "T2"]

    def __init__(self, data_dir, mri_type="Flair", input_size=(256, 256), reshape_input=True):
        self.data_dir = data_dir
        self.mri_type = mri_type
        assert self.mri_type in self.MRI_TYPES
        self.input_size = input_size
        self.reshape_input = reshape_input

        # define transforms
        resize_transform = [transforms.ToPILImage(),
                            transforms.Resize(self.input_size)]
        tensor_transform = [transforms.ToTensor()]
        rgb_transform = [transforms.Grayscale(num_output_channels=3)]

        self._input_image_transforms = transforms.Compose((resize_transform if self.reshape_input else []) +
                                                          rgb_transform +
                                                          tensor_transform)
        self._gt_transforms = transforms.Compose((resize_transform if self.reshape_input else []) +
                                                 tensor_transform)

        # construct self.brain_dirs
        self.brain_dirs = []
        for sub_dir in os.listdir(self.data_dir):
            sub_dir = os.path.join(self.data_dir, sub_dir)
            brain_file_names = os.listdir(sub_dir)
            brain_dirs = [os.path.join(sub_dir, brain_file_name) for brain_file_name in brain_file_names]
            self.brain_dirs += brain_dirs

    def __len__(self):
        return len(self.brain_dirs)
    
    def __getitem__(self, idx):
        brain_dir = self.brain_dirs[idx]

        # find best slice for this brain
        gt_mha_file_path = get_gt_mha_file_path(brain_dir)
        gt_mha_3d = np_from_mha_path(gt_mha_file_path)
        peak_slice = get_most_activated_slice(gt_mha_3d)        
        
        # extract gt_slice
        gt_slice = gt_mha_3d[peak_slice]
        # normalize, to PIL, resize, to tensor
        gt_slice = util.normalize_numpy(gt_slice)
        gt_slice = self._gt_transforms(gt_slice)
        
        # extract mri_slices
        # find path of file for the correct mri type
        mri_type_pattern = ".MR_{mri_type}.".format(mri_type=self.mri_type)
        mri_mha_file_names = [file_name for file_name in os.listdir(brain_dir) if mri_type_pattern in file_name]
        assert len(mri_mha_file_names) == 1
        mri_mha_file_name = mri_mha_file_names[0]
        mri_mha_file_path = os.path.join(brain_dir, mri_mha_file_name)
        # read 3d numpy scan
        mri_mha_3d = np_from_mha_path(mri_mha_file_path)
        # pick only slice with biggest tumor appearance
        mri_slice = mri_mha_3d[peak_slice, :, :]
        # normalize, to PIL, resize, to RGB, to tensor
        mri_slice = util.normalize_numpy(mri_slice)
        mri_slice = self._input_image_transforms(mri_slice)

        return mri_slice, (gt_slice, 1)
