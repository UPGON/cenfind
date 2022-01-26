import argparse
import json
import re
from pathlib import Path

import numpy as np
from cv2 import cv2

from centrack.annotation import Contour
from data import PixelSize, Condition
from detectors import FocusDetector, NucleiStardistDetector


def parse_args():
    parser = argparse.ArgumentParser(description='CCOUNT: Automatic centriole scoring')

    parser.add_argument('dataset', type=Path, help='path to the dataset')
    parser.add_argument('marker', type=str, help='marker to use for foci detection')
    parser.add_argument('-t', '--test', type=int, help='test; only run on the ith image')
    parser.add_argument('-o', '--out', type=Path, help='path for output')

    return parser.parse_args()


def extract_filename(file):
    file_name = file.name
    file_name = file_name.removesuffix(''.join(file.suffixes))
    file_name = file_name.replace('', '')
    file_name = re.sub(r'_(Default|MMStack)_\d-Pos', '', file_name)

    return file_name.replace('', '')


def is_tif(filename):
    _filename = str(filename)
    return _filename.endswith('.tif') and not _filename.startswith('.')


def contrast(data):
    return cv2.convertScaleAbs(data, alpha=255 / data.max())


def image_tint(image, tint):
    """
    Tint a gray-scale image with the given tint tuple
    :param image:
    :param tint:
    :return:
    """
    return (image * tint).astype(np.uint8)


def channels_combine(stack, channels=(1, 2, 3)):
    if stack.shape != (4, 2048, 2048):
        raise ValueError(f'stack.shape')

    stack = stack[channels, :, :]
    stack = np.transpose(stack, (1, 2, 0))

    return cv2.convertScaleAbs(stack, alpha=255 / stack.max())


def nuclei_segment(image, threshold=None):
    """
    Extract the nuclei into contours.
    :param image: the image to segment
    :param threshold: if specified, use it instead of the derived.
    :return: the list of contours detected
    """

    # Define a large blurring kernel (1/16 of the image width)
    _data = image.contrast().data
    image_w, image_h = _data.shape[-2:]
    kernel_size = image_w // 32
    if kernel_size % 2 == 0:
        kernel_size += 1

    nuclei_blurred = (_data
                      .blur_median(ks=kernel_size)
                      .blur_median(ks=kernel_size))

    if threshold:
        threshold_otsu = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        ret, nuclei_thresh = cv2.threshold(nuclei_blurred, 0, 255, threshold_otsu)
    else:
        ret, nuclei_thresh = cv2.threshold(nuclei_blurred, threshold, 255, cv2.THRESH_BINARY)

    nuclei_contours, hierarchy = cv2.findContours(nuclei_thresh, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    nuclei_contours = [Contour(c, idx=c_id, label='nucleus', confidence=-1)
                       for c_id, c in enumerate(nuclei_contours)]

    return nuclei_contours


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


def get_markers(markers, sep='+'):
    """
    Convert a '+'-delimited string into a list and prepend the DAPI
    :param markers:
    :param sep: delimiter character
    :return: List of markers
    """
    markers_list = markers.split(sep)
    if 'DAPI' not in markers_list:
        markers_list.insert(0, 'DAPI')
    return markers_list


def condition_from_filename(file_name, pattern):
    """
    Extract parameters of dataset.
    :param file_name:
    :param pattern: must contain 4 groups, namely: genotype, treatment, markers, replicate
    :return: Condition object
    """

    pat = re.compile(pattern)
    matched = re.match(pat, file_name)
    if matched is not None:
        genotype, treatment, markers, replicate = matched.groups()
    else:
        raise re.error('no matched element')
    markers_list = get_markers(markers)
    return Condition(genotype=genotype,
                     treatment=treatment,
                     markers=markers_list,
                     replicate=replicate,
                     pixel_size=PixelSize(.1025, 'um'))


def extract_centriole(data):
    """
    Extract the centrioles from the channel image.
    :param data:
    :return: List of Points
    """
    focus_detector = FocusDetector(data, 'Centriole')
    return focus_detector.detect()


def extract_nuclei(data):
    """
    Extract the nuclei from the nuclei image.
    :param data:
    :return: List of Contours.
    """
    nuclei_detector = NucleiStardistDetector(data, 'Nucleus')
    return nuclei_detector.detect()


def prepare_background(nuclei, foci):
    """
    Create a BGR image from the nuclei (gray) and the centriole (red) channels.
    :param nuclei:
    :param foci:
    :return:
    """
    background = cv2.cvtColor(contrast(nuclei), cv2.COLOR_GRAY2BGR)
    foci_bgr = np.zeros_like(background)
    foci_bgr[:, :, 2] = contrast(foci)
    return cv2.addWeighted(background, .5, foci_bgr, 1, 1)


def draw_annotation(background, res, foci_detected=None, nuclei_detected=None):
    """
    Draw the assigned centrioles on the background image.
    :param background:
    :param res:
    :param foci_detected:
    :param nuclei_detected:
    :return: BGR image displaying the annotation
    """
    for c in foci_detected:
        c.draw(background)

    for n in nuclei_detected:
        n.draw(background)

    for c, n in res:
        start = c.centre
        end = n.centre.centre
        cv2.line(background, start, end, (0, 255, 0), 3, lineType=cv2.FILLED)

    return background
