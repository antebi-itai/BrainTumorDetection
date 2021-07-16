import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import util
import random
import SimpleITK as sitk
import numpy as np
import cv2

tumor_type2name = {0: "no", 1: "yes"}
tumor_name2type = {"no": 0, "yes": 1}


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
    """

    def __init__(self, data_dir, input_size=(256, 256), reshape_input=True, **kwargs):
        self.data_dir = data_dir
        self.tumor_dirs = os.listdir(self.data_dir)
        self.input_size = input_size
        self.reshape_input = reshape_input
        assert all([tumor_dir.endswith("_tumor") for tumor_dir in self.tumor_dirs])

        # define transforms
        self._input_transforms = transforms.Compose(([transforms.Resize(self.input_size)] if self.reshape_input else []) +
                                                    [transforms.Grayscale(num_output_channels=3),
                                                     transforms.ToTensor()])

        # parse data from directory to convenient data structures
        self.tumor_image_paths = []
        self.tumor_types = []
        for tumor_dir in self.tumor_dirs:
            tumor_name = tumor_dir.replace("_tumor", "")
            if tumor_name != "no": tumor_name = "yes"
            tumor_type = tumor_name2type[tumor_name]
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
        return tumor_image_input, (float('nan'), tumor_label)

    def make_weights_for_balanced_classes(self):
        nclasses = len(tumor_type2name)

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
    def __init__(self, ref_image, occlusion_size=(50, 50)):
        self.ref_image = ref_image
        self.occlusion_size = occlusion_size

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


def get_most_middle_no_tumor_index(gt_3d):
    num_slices = gt_3d.shape[0]
    middle_slice = num_slices // 2

    zero_gt_slice_indices = np.argwhere(np.count_nonzero(gt_3d, axis=(1, 2)) == 0).squeeze()
    if zero_gt_slice_indices.size == 0:
        raise RuntimeError("Did not find any no-tumor slice in GT")
    most_middle_no_tumor_index = zero_gt_slice_indices[abs(zero_gt_slice_indices - middle_slice).argmin()]
    return most_middle_no_tumor_index


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
        rgb_transform = [transforms.Grayscale(num_output_channels=3)]
        tensor_transform = [transforms.ToTensor()]

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
        return 2 * len(self.brain_dirs)
    
    def __getitem__(self, idx):
        brain_dir = self.brain_dirs[idx // 2]
        with_tumor = idx % 2

        # find best slice for this brain
        gt_mha_file_path = get_gt_mha_file_path(brain_dir)
        gt_3d = np_from_mha_path(gt_mha_file_path)
        if with_tumor:
            # pick only slice with biggest tumor appearance
            slice_index = get_most_activated_slice(gt_3d)
        else:
            # pick slice closest to middle slice
            slice_index = get_most_middle_no_tumor_index(gt_3d)
        
        # extract gt_slice
        gt_slice = gt_3d[slice_index]
        # normalize, to PIL, resize, to tensor
        gt_slice = np.uint8(gt_slice)
        gt_slice = self._gt_transforms(gt_slice)
        gt_slice = (gt_slice.squeeze() * 256).type(torch.uint8)
        
        # extract mri_slice
        # find path of file for the correct mri type
        mri_type_pattern = ".MR_{mri_type}.".format(mri_type=self.mri_type)
        mri_mha_file_names = [file_name for file_name in os.listdir(brain_dir) if mri_type_pattern in file_name]
        assert len(mri_mha_file_names) == 1
        mri_mha_file_name = mri_mha_file_names[0]
        mri_mha_file_path = os.path.join(brain_dir, mri_mha_file_name)
        # read 3d numpy scan
        mri_3d = np_from_mha_path(mri_mha_file_path)
        # pick only correct slice
        mri_slice = mri_3d[slice_index, :, :]
        # normalize, to PIL, resize, to RGB, to tensor
        mri_slice = np.uint8(cv2.normalize(mri_slice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        mri_slice = self._input_image_transforms(mri_slice)

        return mri_slice, (gt_slice, with_tumor)

    def make_weights_for_balanced_classes(self):
        nclasses = len(self)
        return [1] * nclasses
