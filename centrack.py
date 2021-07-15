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

from utils import image_8bit_contrast


def nuclei_extract(nuclei, nucleus_area_min=1000):
    """
    Extract the nuclei into contours.
    :param nuclei:
    :param nucleus_area_min:
    :return:
    """

    # Apply the gaussian blurring filter with large kernel (1/16 of the image width)
    image_w, image_h = nuclei.shape[-2:]
    kernel_size = image_w // 32 + 1

    blurred = cv2.GaussianBlur(nuclei, (kernel_size, kernel_size), 0)

    # Threshold the image
    threshold_mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ret, nuclei_thresh = cv2.threshold(blurred, 0, 255, threshold_mode)

    # Find the contours
    contours, hierarchy = cv2.findContours(nuclei_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nuclei_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > nucleus_area_min]

    return nuclei_thresh, nuclei_contours


def centrioles_detect(image, threshold_foci, distance_min, kernel_size=None):
    """
    Detect the foci above a defined threshold and a minimal inter-peak distance.
    :param image:
    :param threshold_foci:
    :param distance_min:
    :param kernel_size:
    :return:
    """
    image[image < threshold_foci] = 0

    if kernel_size:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

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


def args_parse():
    parser = argparse.ArgumentParser(description='Centriole detector')

    parser.add_argument('--path', type=str,
                        help='the path to the dataset folder')
    parser.add_argument('--nucleus-area-min', type=int,
                        help='the maximal area in pixel for a nucleus')
    parser.add_argument('--channel', type=int,
                        help='Set the channel reference to use [0-4]')
    parser.add_argument('--threshold-foci', type=int,
                        help='Threshold for focus brightness (16bit depth)')
    parser.add_argument('--distance', type=int,
                        help='Minimal distance between two foci')
    parser.add_argument('--kernel-size', type=int,
                        help='Kernel size to smooth the centriole channel')

    return vars(parser.parse_args())


def main(args):
    # pdb.set_trace()
    path_dataset = Path(args['path'])
    dataset_name = path_dataset.name
    path_projections = path_dataset / 'projected'
    path_out = path_dataset / 'out'

    distance_min = args['distance']
    threshold_foci = args['threshold_foci']
    nucleus_area_min = args['nucleus_area_min']

    # Collect the ome.tiff files
    files = sorted(tuple(file for file in path_projections.iterdir()
                         if file.name.endswith('.tif')
                         if not file.name.startswith('.')))
    file = files[0]
    print('Loading')
    data = tf.imread(file)

    # Extract the DAPI channel
    # nuclei = data[0, :, :]
    # nuclei = cv2.convertScaleAbs(nuclei, alpha=255 / nuclei.max())
    # nuclei_threshold, nuclei_contours = nuclei_extract(nuclei, nucleus_area_min)
    #
    # nuclei_8bit_bgr = cv2.cvtColor(nuclei, cv2.COLOR_GRAY2BGR)
    # nuclei_annotated = cv2.drawContours(nuclei_8bit_bgr, nuclei_contours, -1, (0, 255, 0), cv2.FILLED)
    # cv2.imwrite(str(path_out / f'{dataset_name}_1_nuclei.png'), nuclei_annotated)

    centrioles = data[args['channel'], :, :]
    centrioles_8bit = image_8bit_contrast(centrioles)
    cv2.imwrite(str(path_out / f'{dataset_name}_2_centrioles.png'),
                cv2.bitwise_not(centrioles_8bit))

    foci_coords = centrioles_detect(centrioles, threshold_foci, distance_min, kernel_size=3)

    for i, (r, c) in enumerate(foci_coords):
        cv2.drawMarker(centrioles_8bit, (c, r), 255, cv2.MARKER_CROSS, 10)
        cv2.putText(img=centrioles_8bit, text=str(i), org=(c + 10, r + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=2, color=128)

    cv2.imwrite(str(path_out / f'{dataset_name}_2_centrioles_peaks_T{threshold_foci}.png'),
                cv2.bitwise_not(centrioles_8bit))


if __name__ == '__main__':
    args = args_parse()
    main(args)
