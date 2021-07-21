import operator
from pathlib import Path
import argparse

import cv2
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import img_as_float

import pdb


def centrioles_detect(image, threshold_foci, distance_min):
    """
    Detect the foci above a defined threshold and a minimal inter-peak distance.
    :param image:
    :param threshold_foci:
    :param distance_min:
    :return:
    """

    image[image < threshold_foci] = 0

    centrioles_float = img_as_float(image.copy())
    foci_coords = peak_local_max(centrioles_float, min_distance=distance_min)

    return foci_coords


def centriole_assign(nuclei, centrioles):
    pass
    # # Assign centriole to their nearest neighbour
    # contours_nearest = []
    #
    # for c, centriole in enumerate(foci_coords):
    #     r_c, c_c = centriole
    #
    #     distance_to_nuclei = []
    #
    #     for n, nucleus in enumerate(contours_nuclei):
    #         dist = cv2.pointPolygonTest(nucleus, (c_c, r_c), True)
    #         distance_to_nuclei.append((c, n, int(abs(dist))))
    #
    #     contours_nearest.append(min(distance_to_nuclei, key=operator.itemgetter(-1)))
    #
    # annotation_layer = np.zeros_like(nuclei_8bit)
    # structures = np.zeros_like(nuclei_8bit_bgr)
    #
    # for el in contours_nearest:
    #     centriole_id, nucleus_id, dist = el
    #
    #     if abs(dist) > 200:
    #         print('orphan', dist)
    #         continue
    #
    #     r_c, c_c = foci_coords[centriole_id]
    #     nucleus = contours_nuclei[nucleus_id]
    #
    #     nucleus_moments = cv2.moments(nucleus)
    #
    #     r_n, c_n = int(nucleus_moments['m01'] / nucleus_moments['m00']), int(nucleus_moments['m10'] / nucleus_moments['m00'])
    #
    #     centriole_centre = (c_c, r_c)
    #     nucleus_centre = (c_n, r_n)
    #
    #     cv2.drawMarker(annotation_layer, centriole_centre, 255, cv2.MARKER_CROSS)  # centrioles cv2.MARKER_CROSS
    #     cv2.circle(annotation_layer, (c_n, r_n), 10, 255, -1)  # nuclei
    #     cv2.arrowedLine(annotation_layer, centriole_centre, nucleus_centre, 255, 2)  # assignment
    #
    # # cv2.addWeighted(annotation_layer, alpha, nuclei_bgr, alpha, 0, final)
    # structures[:, :, 0] = nuclei_8bit
    # structures[:, :, 1] = annotation_layer
    # structures[:, :, 2] = centrioles_8bit
    # cv2.imwrite(str(path_out / f'{dataset_name}_3_assigned.png'), structures)
