import argparse
import re
from pathlib import Path

import numpy as np
from cv2 import cv2

from data import PixelSize, Condition


def parse_args():
    parser = argparse.ArgumentParser(
        description='CCOUNT: Automatic centriole scoring')

    parser.add_argument('dataset', type=Path, help='path to the dataset')
    parser.add_argument('marker', type=str,
                        help='marker to use for foci detection')
    parser.add_argument('-t', '--test', type=int,
                        help='test; only run on the ith image')
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


def channels_combine(stack, channels=(1, 2, 3)):
    if stack.shape != (4, 2048, 2048):
        raise ValueError(f'stack.shape')

    stack = stack[channels, :, :]
    stack = np.transpose(stack, (1, 2, 0))

    return cv2.convertScaleAbs(stack, alpha=255 / stack.max())


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
        cv2.line(background, start, end, (0, 255, 0), thickness=2,
                 lineType=cv2.FILLED)

    return background
