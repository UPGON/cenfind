import json
from pathlib import Path

import cv2
import numpy as np
import tifffile as tf
from skimage import img_as_float
from skimage.feature import peak_local_max

from matplotlib import pyplot as plt


def image_8bit_contrast(image):
    return cv2.convertScaleAbs(image, alpha=255 / image.max())


def file_read(path):
    """
    Read an ome tiff file.
    :param path:
    :return: Numpy array
    """
    print('Reading the raw ome tiff file')
    fov = tf.imread(path)

    dimensions = fov.shape

    if dimensions[0] > dimensions[1]:
        _target_dims = list(dimensions)
        _target_dims[0], _target_dims[1] = _target_dims[1], _target_dims[0]
        target_dims = tuple(_target_dims)
        fov = fov.flatten().reshape(target_dims)

    return fov


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


def coords2mask(foci_coords, shape):
    mask = np.zeros(shape, np.uint8)
    for r, c in foci_coords:
        mask[r, c] = 255

    return mask


def centrosomes_segment(foci_mask):
    """
    Compute the contours of neighbouring foci.
    :param foci_coords:
    :return: list of contours
    """
    # convert to np.float32
    mask = np.float32(foci_mask)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(mask, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    A = mask[label.ravel() == 0]
    B = mask[label.ravel() == 1]

    return A, B


def cell_segment(image):
    """
    Segment the cell based on the nuclei
    :param image:
    :return: a pixel-wise classification
    """
    ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = markers.astype('int32')
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 255]

    return image


def cnt_centre(contour):
    """
    Compute the centre of a contour
    :param contour:
    :return: the coordinates of the contour
    """
    moments = cv2.moments(contour)

    c_x = int(moments['m10'] / moments['m00'])
    c_y = int(moments['m01'] / moments['m00'])

    return c_x, c_y


def main():
    path_root = Path('/Volumes/work/datasets')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'
    fov_name = f'{dataset_name}_001_000.tif'

    path_fov = path_root / dataset_name / 'raw' / fov_name
    path_projected = path_root / dataset_name / 'projected' / fov_name

    # data = file_read(path_fov)
    projected = tf.imread(path_projected, key=range(4))

    centrioles = channel_extract(projected, 1)
    centrioles = cv2.GaussianBlur(centrioles, (3, 3), sigmaX=0)
    centrioles = cv2.medianBlur(centrioles, 3)
    nuclei = channel_extract(projected, 0)

    ret1, thresh = cv2.threshold(centrioles, 0, 65536, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('out/centriole_thresh.png', thresh)
    foci_coords = centrioles_detect(centrioles, threshold_foci=500, distance_min=2)
    # foci_mask = coords2mask(foci_coords, centrioles.shape)
    image = np.zeros((2048, 2048, 3), np.uint8)
    image[:, :, 0] = image_8bit_contrast(nuclei)
    image[:, :, 1] = image_8bit_contrast(centrioles)

    for x, y in foci_coords:
        cv2.circle(image, (y, x), 20, (255, 255, 255), 1)
        # cv2.drawMarker(image, (y, x), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2,
        #                line_type=1)

    cv2.imwrite('out/artefact.png', image)

    cells_contours = cell_segment(nuclei)
    # centrosomes_contours = centrosomes_segment(foci_coords)

    results = []

    for n, cnt_cell in enumerate(cells_contours):
        for c, cnt_cm in enumerate(centrosomes_contours):
            centre = cnt_centre(cnt_cm)
            if cv2.pointPolygonTest(cnt_cell, centre, measureDist=False) < 0:
                for f, focus in enumerate(foci_coords):
                    if cv2.pointPolygonTest(cnt_cm, focus, measureDist=False) < 0:
                        results.append((n, c, f))


if __name__ == '__main__':
    main()
