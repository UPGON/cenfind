import albumentations as alb
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

transforms = alb.Compose([
    alb.ShiftScaleRotate(scale_limit=0.),
    alb.Flip(),
])


def compute_focal_loss_weights(scores, targets, gamma):
    probs = torch.sigmoid(scores)
    oneminuspt = probs * (1 - targets) + (1 - probs) * targets
    weights = oneminuspt ** gamma
    return weights


def preprocess_image(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    data = data + np.random.randn(*data.shape) * np.random.rand(1) * 0.1
    scale = 0.5 + 0.5 * np.random.rand(1)
    bias = np.random.rand(1)
    data = data * scale + bias

    return data


def compute_mindist_map(mask, annotation):
    y, x = np.unravel_index(np.arange(mask.size), mask.shape)
    pts = np.concatenate((x.astype(float).reshape((-1, 1)),
                          y.astype(float).reshape((-1, 1))),
                         axis=1)
    dist = pairwise_distances(pts, annotation)
    min_dist = np.min(dist, axis=1).reshape(mask.shape)

    return min_dist


def compute_labelmap(mask, min_dist):
    """
    Compute the distance map to the nearest foci
    :param annotation:
    :param mask_shape:
    :return:
    """
    labelmap = mask.copy()
    labelmap[min_dist < 1] = 1

    return labelmap


def compute_weight(mask, min_dist):
    weight = np.ones_like(mask)
    weight[min_dist < 5] = 0
    weight[min_dist < 1] = 1000

    return weight


def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines
