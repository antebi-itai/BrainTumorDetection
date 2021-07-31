import numpy as np


def coords2idx(pt, image):
    return pt[0] * image.shape[1] + pt[1]


def idx2coords(idx, image):
    i = idx // image.shape[1]
    j = idx % image.shape[1]

    return i, j


def occlude_image(image, idx, occlusion_size=(60, 60)):
    i, j = idx2coords(idx, image)

    h = occlusion_size[0] // 2
    w = occlusion_size[1] // 2

    left = max(j - w, 0)
    up = max(i - h, 0)
    right = min(j + w, image.shape[2])
    down = min(i + h, image.shape[1])

    image[:, up:down, left:right] = 0

    return image


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_t = move2cpu(im_t)
    if len(im_t.shape) == 3:
        im_t = np.transpose(im_t, (1, 2, 0))
    im_np = np.clip(np.round(im_t * 255.0), 0, 255)
    return im_np.astype(np.uint8)

