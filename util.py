import torch


def idx2coords(idx, image):
    i = idx // image.shape[1]
    j = idx % image.shape[1]

    return i, j


def occlude_image(image, idx, occlusion_size=(60, 60)):
    i, j = idx2coords(idx)

    h = occlusion_size[0] // 2
    w = occlusion_size[1] // 2

    left = max(j - w, 0)
    up = max(i - h, 0)
    right = min(j + w, image.shape[1])
    down = min(i + h, image.shape[0])

    image[up:down, left:right] = 0

    return image
