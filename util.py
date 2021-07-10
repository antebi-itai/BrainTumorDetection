import cv2

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


def produce_visual_heatmaps(image, heatmap):
    gray_heatmap = normalize_numpy(heatmap.cpu().numpy())
    colorful_heatmap = cv2.cvtColor(cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = 0.5 * colorful_heatmap + 0.5 * normalize_numpy(image.cpu().numpy())

    return gray_heatmap, overlay


def normalize_numpy(np_image):
    return (((np_image - np_image.min()) / (np_image.max() - np_image.min())) * 255).astype("uint8")
