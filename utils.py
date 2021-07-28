from pathlib import Path
import json
from matplotlib import pyplot as plt
import cv2
import tifffile as tf
import numpy as np

import pdb


def channels_combine(stack, channels=(1, 2, 3)):
    if stack.shape != (4, 2048, 2048):
        raise ValueError(f'stack.shape')

    stack = stack[channels, :, :]
    stack = np.transpose(stack, (1, 2, 0))

    return cv2.convertScaleAbs(stack, alpha=255 / stack.max())


def label_mask_write(dest, labels):
    """
    Write a visualisation of the labels.
    :param dest:
    :param labels:
    :return:
    """
    labels_vis = 255 * (labels / labels.max())
    cv2.imwrite(str(dest), labels_vis)


def mask_create_from_contours(mask, contours):
    """
    Label each blob using connectedComponents.
    :param mask: the black image to draw the contours
    :param contours: the list of contours
    :return: the mask with each contour labelled with different numbers.
    """
    cv2.drawContours(mask, contours, -1, 255, -1)
    _, labels = cv2.connectedComponents(mask)
    return labels



def labelbox_annotation_load(path_annotation, image_name):
    with open(path_annotation, 'r') as file:
        annotation = json.load(file)
    image_labels = [image for image in annotation if image['External ID'] == image_name]

    return image_labels[0]['Label']['objects']


def label_coordinates(label):
    return label['point'].values()


def image_8bit_contrast(image):
    return cv2.convertScaleAbs(image, alpha=255 / image.max())


def markers_from(dataset_name, marker_sep='+'):
    """
    Extract the markers' name from the dataset string.
    The string must follows the structure `<genotype>_marker1+marker2`
    It append the DAPI at the beginning of the list.

    :param marker_sep:
    :param dataset_name:
    :return: a dictionary of markers
    """

    markers = dataset_name.split('_')[-2].split(marker_sep)

    if 'DAPI' not in markers:
        markers = ['DAPI'] + markers

    return {k: v for k, v in enumerate(markers)}


def channel_extract(stack, channel_id):
    """
    Extract a channel and apply a projection.
    :param stack: 3D array
    :return: 2D array for the channel
    """
    return stack[channel_id, :, :]


def coords2mask(foci_coords, shape):
    mask = np.zeros(shape, np.uint8)
    for r, c in foci_coords:
        mask[r, c] = 255

    return mask


if __name__ == '__main__':
    path_root = Path('/Volumes/work/datasets/RPE1wt_CEP63+CETN2+PCNT_1')
    path_raw = path_root / 'raw'

    file = path_root / 'projected/RPE1wt_CEP63+CETN2+PCNT_1_000_000.png'
    labels = labelbox_annotation_load(path_root / 'annotation.json', file.name)
    print(0)

    # reshaped = fov_read(path_raw / file.name)
    # profile, projected = sharp_planes(reshaped, 1, 0)
    # projected_rgb = channels_combine(projected)
    # tf.imwrite('/Users/leoburgy/Desktop/test.png', projected[0, :, :])
    print(0)
