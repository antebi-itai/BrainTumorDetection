import torch
from train import train, train_loop
from feature_extractor import FeatureExtractor
import data
from data import OccludedImageGenerator
from network import get_model_and_optim, load_best_state
import wandb
from tqdm import tqdm
from post_process import calc_iou, calc_dice, present_grad_masks, present_all_masks, mask_from_heatmap, get_masks_from_heatmaps
from util import tensor2im
import cv2
import numpy as np
from torch.autograd import backward
wandb.login()


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        for key, value in config.items():
            setattr(self, key, value)

        # unite heat layer and additional layers
        self.heat_layers.append(self.ref_heat_layer)

        # train
        self.train_dataset = eval(self.data_train_class)(data_dir=self.data_train_path, mri_type=self.mri_type, input_size=self.input_size)
        train_weights = self.train_dataset.make_weights_for_balanced_classes()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.train_batch_size, sampler=train_sampler)
        # test
        self.test_dataset = eval(self.data_test_class)(data_dir=self.data_test_path, mri_type=self.mri_type, input_size=self.input_size)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.train_batch_size, shuffle=True)

        self.criterion = torch.nn.functional.cross_entropy
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device)

    def run(self):
        self.train_model()
        self.model_acc = self.eval_model()
        print("Model's accuracy: {}".format(self.model_acc), flush=True)

        extra_dsc = {}
        for layer in self.heat_layers:
            extra_dsc[str(layer)] = []
        cold_mask_dice = []
        grad_mask_dice = []

        for image_num in range(len(self.test_dataset) // 2):
            title = "#{image_num}".format(image_num=image_num)

            # Original image
            original_image, (gt_mask, gt_) = self.test_dataset[image_num * 2 + 1]
            gt_mask = gt_mask > self.gt_threshold
            wandb.log({"{title}/original_image".format(title=title): [wandb.Image(original_image)]})

            # Generate heatmaps
            heatmaps = self.generate_heatmap(original_image, title=title)

            # Calculate masks
            hot_masks, cold_masks = get_masks_from_heatmaps(heatmaps,
                                                            thresh=self.heatmap_threshold,
                                                            smallest_contour_len=self.smallest_contour_len)
            grad_mask = self.get_mask_using_gradient(original_image,
                                                     std_thresh=self.grad_std_thresh,
                                                     kernel_size=self.grad_kernel_size,
                                                     contour_threshold=self.grad_contour_threshold,
                                                     smallest_contour_len=self.smallest_contour_len,
                                                     device=self.device)

            present_all_masks(original_image=original_image, gt_mask=gt_mask, grad_mask=grad_mask,
                          hot_masks=hot_masks, cold_masks=cold_masks, title=title)

            # calculate IOU
            gt_mask_cpu = gt_mask.cpu().numpy()
            for layer in self.heat_layers:
                temp = extra_dsc[str(layer)]
                temp.append(calc_iou(gt_mask_cpu, cold_masks[str(layer)]))
                wandb.log({"DSC/{title}/{layer}".format(title=title, layer=str(layer)): temp[-1]})

        # log hyperparameters
        for layer in self.heat_layers:
            temp = extra_dsc[str(layer)]
            wandb.log({"DSC/avg_dsc/{layer}".format(layer=str(layer)): sum(temp) / len(temp)})

        self.log_hyperparameters(additional_atributes=['model_acc'])
        return cold_mask_dice, grad_mask_dice

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion,
              optimizer=self.optimizer, train_loader=self.train_loader, test_loader=self.test_loader,
              epochs=self.epochs, device=self.device)
        load_best_state(self.model, self.optimizer)

    """
    TODO: Doc
    """
    def eval_model(self):
        accuracies = []
        weights = []
        for (test_images, (test_tumor_segmentations, test_tumor_types)) in self.test_loader:
            # test accuracy of batch
            accuracy = train_loop(model=self.model, criterion=self.criterion,
                                  optimizer=self.optimizer, device=self.device,
                                  images=test_images, tumor_types=test_tumor_types, mode="Test")
            accuracies.append(accuracy)
            weights.append(test_images.size(0))
        test_accuracy = sum([accuracy * weight for accuracy, weight in zip(accuracies, weights)]) / sum(weights)
        return round(test_accuracy, 2)

    """
    TODO: Doc
    """
    def generate_heatmap(self, original_image, title=""):
        """
        Generate heat maps from image, based on experiment's model and heat layers

        :param original_image: image from which to create the heat maps
        :return: heatmaps: dictionary containing descriptive keys (L`layer`C`channel) and corresponding heatmap.
                           ordered in the same manner as the heat layers in the configuration
        """
        occluded_loader = torch.utils.data.DataLoader(OccludedImageGenerator(original_image, occlusion_size=self.occlusion_size),
                                                      batch_size=self.heatmap_batch_size, shuffle=False)

        # Start extracting features
        fe = FeatureExtractor(model=self.model, device=self.device)

        # Hook heat layers
        for heat_layer in self.heat_layers:
            fe.plug_layer(heat_layer)

        # Forward pass over original image
        self.model.eval()
        original_image = original_image.to(self.device).unsqueeze(0)
        height, width = original_image.shape[2], original_image.shape[3]

        self.model(original_image)
        feature_layers = fe.flush_layers()

        # Hook the channel with max value
        for layer_pos, (_, feature_layer) in zip(self.heat_layers, feature_layers.items()):
            if len(feature_layer[0].shape) == 4:
                # Set the most 'activated channel' as reference
                channel = torch.argmax(torch.sum(feature_layer[0], dim=(2, 3))).item()  # Sum over spatial dimensions
                fe.plug_activation(layer_pos, channel)
            else:
                fe.plug_layer(layer_pos)

        # Forward pass over all occlusions of an image
        print("Creating heatmap...", flush=True)
        with torch.no_grad():
            for idx, images in tqdm(occluded_loader):
                occluded_images = images.to(device=self.device)
                self.model(occluded_images)

        activations = fe.flush_activations()
        layers = fe.flush_layers()

        heatmaps = {}
        overlay_heatmaps = {}
        # Generate Convolutional heatmaps
        for channel, heatmap_array in activations.items():
            gray_heatmap = torch.cat(heatmap_array).reshape((height, width)).cpu().numpy()
            gray_heatmap = np.uint8(cv2.normalize(gray_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[channel] = gray_heatmap
            np_original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            np_original_image = np.uint8(cv2.normalize(np_original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            overlay_heatmaps[channel] = 0.5 * colorful_heatmap + 0.5 * np_original_image
            
        # Generate Linear layers heatmaps
        for layer, heatmap_array in layers.items():
            heatmap = torch.cat(heatmap_array).reshape((height, width, -1))
            correlated_heatmap = heatmap[:, :, 1] - heatmap[:, :, 0]
            gray_heatmap = correlated_heatmap.cpu().numpy()
            gray_heatmap = np.uint8(cv2.normalize(gray_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmaps[layer] = gray_heatmap
            np_original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            np_original_image = np.uint8(cv2.normalize(np_original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            overlay_heatmaps[layer] = 0.5 * colorful_heatmap + 0.5 * np_original_image

        # Log heatmaps
        wandb.log({"{title}/grayscale_heatmaps".format(title=title):
                       [wandb.Image(gray_heatmap, caption=channel) for channel, gray_heatmap in heatmaps.items()]})
        wandb.log({"{title}/overlay_heatmaps".format(title=title):
                       [wandb.Image(overlay_heatmap, caption=channel) for channel, overlay_heatmap in overlay_heatmaps.items()]})

        return heatmaps

    def get_mask_using_gradient(self, original_image, std_thresh=2, kernel_size=30, contour_threshold=0.5,
                                smallest_contour_len=30, device="cuda"):
        # Move to device
        images, tumor_types = original_image.unsqueeze(0).to(device=device), torch.zeros(1).to(device=device)
        images.requires_grad = True
        # Run the model on the input image
        pred_tumors_scores = self.model(images)
        # Calculate the loss for this image
        loss = torch.nn.functional.cross_entropy(pred_tumors_scores, tumor_types.to(dtype=torch.long))
        # Backprop the gradients to the image
        self.optimizer.zero_grad()
        backward(loss)

        # gradients
        grad_map = images.grad.squeeze().sum(axis=0)
        # threshold using std
        grad_map = (grad_map > (grad_map.mean() + (grad_map.std() * std_thresh))).float()
        # blur
        grad_map = torch.nn.functional.conv2d(input=grad_map.view(torch.Size([1, 1]) + grad_map.shape),
                                              weight=(torch.ones(1, 1, kernel_size,
                                                                 kernel_size) / kernel_size ** 2).cuda(),
                                              padding='same').squeeze()
        # normalized numpy
        grad_map = np.uint8(cv2.normalize(tensor2im(grad_map), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        # contours
        grad_mask = mask_from_heatmap(grad_map, thresh=contour_threshold, smallest_contour_len=smallest_contour_len)

        return grad_mask

    def log_hyperparameters(self, additional_atributes):
        table = wandb.Table(columns=self.attributes_to_log)
        table.add_data(*[str(getattr(self, attribute)) for attribute in self.attributes_to_log])
        wandb.log({"hyper_parameters": table})
