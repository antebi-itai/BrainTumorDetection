import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import wandb


def filter_contours(contours, smallest_contour_len=30):
    filter_contours = []
    for contour in contours:
        x, y = contour.T[:, 0, :]
        # contours are complex and do not touch the boundaries
        if (len(contour) > smallest_contour_len) and \
                ((x.min() != 0) and (y.min() != 0) and (x.max() != 255) and (y.max() != 255)):
            filter_contours.append(contour)
    return filter_contours


def smoothened_contours(contours):
    smoothened_contours = []
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1.0, per=1, quiet=2)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
    return smoothened_contours


def mask_from_heatmap(image, thresh=0.9, smallest_contour_len=30):
    # binary mask according to threshold
    ret, thresh_img = cv2.threshold(src=image, thresh=thresh * 255, maxval=255, type=cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # filter out irrelevant contours, and smooth the rest
    contours = filter_contours(contours, smallest_contour_len=smallest_contour_len)
    contours = smoothened_contours(contours=contours)
    # draw only filled contours
    mask = np.zeros(image.shape)
    #for contour in contours:
    cv2.fillPoly(img=mask, pts=contours, color=255)
    return mask


def get_masks_from_heatmaps(heatmaps, thresh=0.9, smallest_contour_len=30):
    hot_masks = {}
    cold_masks = {}
    for channel, heatmap in heatmaps.items():
        hot_masks[channel]  = mask_from_heatmap(image=heatmap, thresh=thresh, smallest_contour_len=smallest_contour_len)
        cold_masks[channel] = mask_from_heatmap(image=255-heatmap, thresh=thresh, smallest_contour_len=smallest_contour_len)
    wandb.log({"heatmaps/hot_masks":  [wandb.Image(hot_mask,  caption=channel) for channel, hot_mask in hot_masks.items()]})
    wandb.log({"heatmaps/cold_masks": [wandb.Image(cold_mask, caption=channel) for channel, cold_mask in cold_masks.items()]})
    return hot_masks, cold_masks
