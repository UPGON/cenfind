from pathlib import Path
import json
from matplotlib import pyplot as plt
import cv2
import tifffile as tf
import numpy as np

import pdb


def sharp_planes(array, reference_channel, threshold):
    """
    Compute the sharpness of the planes and max-project planes above threshold.

    Arguments
    ---------
    array (4D Numpy array): the stack to max-project
    threshold (integer): std threshold above which plane is considered sharp

    Return
    ------
    array (3D): the stack containing the max-projection of each channel
    """
    # pdb.set_trace()
    profile = array[reference_channel, :, :, :].std(axis=(1, 2))

    if any(plane > threshold for plane in profile):
        projected = array[:, profile > threshold, :, :].max(axis=1)
    else:
        projected = array.max(axis=1)

    print(profile.shape, projected.shape)

    return profile, projected


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

    return list(zip(range(len(markers)), markers))


def channel_help(stack, markers_map):
    """
    Plot the intensity of the channels along the depth
    :return: fig, axs
    """
    depth_n = stack.shape[1]
    fig, axs = plt.subplots(ncols=len(markers_map), figsize=(20, 5))

    for i, name in markers_map:
        ax = axs[i]
        profile, projected = sharp_planes(stack,
                                          reference_channel=i,
                                          threshold=0)
        ax.plot(profile, 'k')
        ax.hlines(profile.mean(), 0, depth_n, colors='k')
        ax.set_title(f"{name}; {int(profile.mean())}")

    return fig, axs


def fov_read(path):
    fov = tf.imread(path)

    dimensions = fov.shape

    if dimensions[0] > dimensions[1]:
        _target_dims = list(dimensions)
        _target_dims[0], _target_dims[1] = _target_dims[1], _target_dims[0]
        target_dims = tuple(_target_dims)
        fov = fov.flatten().reshape(target_dims)

    return fov


def channels_combine(stack, channels=(1, 2, 3)):

    if stack.shape != (4, 2048, 2048):
        raise ValueError(f'stack.shape')

    stack = stack[channels, :, :]
    stack = np.transpose(stack, (1, 2, 0))

    return cv2.convertScaleAbs(stack, alpha=255 / stack.max())


def labelbox_annotation_load(path_annotation, image_name):
    with open(path_annotation, 'r') as file:
        annotation = json.load(file)
    image_labels = [image for image in annotation if image['External ID'] == image_name]

    return image_labels[0]['Label']['objects']


def label_coordinates(label):
    return label['point'].values()


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