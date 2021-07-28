import json
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
import tifffile as tf
from skimage import img_as_float
from skimage.feature import peak_local_max

from utils import labelbox_annotation_load, label_coordinates

from matplotlib import pyplot as plt


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


def nuclei_segment(nuclei, dest=None, threshold=None):
    """
    Extract the nuclei into contours.
    :param nuclei:
    :param nucleus_area_min:
    :return:
    """

    # Define a large blurring kernel (1/16 of the image width)
    image_w, image_h = nuclei.shape[-2:]
    kernel_size = image_w // 32
    if kernel_size % 2 == 0:
        kernel_size += 1

    nuclei_blurred = cv2.medianBlur(nuclei, kernel_size)
    nuclei_blurred = cv2.medianBlur(nuclei_blurred, kernel_size)

    if threshold:
        threshold_mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        ret, nuclei_thresh = cv2.threshold(nuclei_blurred, 0, 255, threshold_mode)
    else:
        ret, nuclei_thresh = cv2.threshold(nuclei_blurred, threshold, 255, cv2.THRESH_BINARY)

    nuclei_contours, hierarchy = cv2.findContours(nuclei_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if dest:
        cv2.imwrite(str(dest / 'nuclei_blurred.png'), nuclei_blurred)
        cv2.imwrite(str(dest / 'nuclei_thresh.png'), nuclei_thresh)

    return nuclei_contours


def cell_segment(image):
    """
    Segment the cell based on the nuclei
    :param image:
    :return: a pixel-wise classification
    """
    # ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = markers.astype('int32')
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]

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


def foci_detect(centrioles, dest=None, factor=4):
    """
    Apply median, gaussian blur and relative thresholding.
    :return: list of foci coordinates
    """

    centrioles_8bit = image_8bit_contrast(centrioles)
    mean = centrioles_8bit.mean()
    threshold = factor * mean
    ret1, centrioles_thresh = cv2.threshold(centrioles_8bit, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    centrioles_thresh = cv2.morphologyEx(centrioles_thresh, cv2.MORPH_OPEN, kernel)

    if dest:
        cv2.imwrite(str(dest / 'centrioles_thresh.png'), centrioles_thresh)

    return centrioles_thresh


def centrosomes_box(centrioles_threshold, dest=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    centrosomes = cv2.morphologyEx(centrioles_threshold, op=cv2.MORPH_DILATE, kernel=kernel, iterations=5)
    centrosomes_contours, hierarchy = cv2.findContours(centrosomes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centrosomes_bboxes = []
    for c_id, cnt in enumerate(centrosomes_contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        centrosomes_bboxes.append(np.int0(box))
    if dest:
        cv2.imwrite(str(dest / 'centrosomes.png'), centrosomes)
    return centrosomes_bboxes


def main():
    path_root = Path('/Volumes/work/datasets')

    # dataset_name = '20210709_RPE1_deltS6_Lentis_HA-DM4_B3_pCW571_48hDOX_rCep63_mHA_gCPAP_1'
    # fov_name = f'{dataset_name}_MMStack_Default_max.ome.tif'
    # U2OS_CEP63+SAS6+PCNT_1
    # RPE1wt_CEP152+GTU88+PCNT_1
    # RPE1wt_CEP63+CETN2+PCNT_1
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'
    # fov_name = f'{dataset_name}_000_000_max.ome.tif'
    fov_name = f'{dataset_name}_000_000_max.ome.tif'

    path_projected = path_root / f'{dataset_name}' / 'projections' / fov_name

    path_out = path_root / dataset_name / 'out'
    path_out.mkdir(exist_ok=True)

    # Data loading
    projected = tf.imread(path_projected, key=range(4))
    c, w, h = projected.shape
    # Segment nuclei
    nuclei_raw = channel_extract(projected, 0)
    nuclei_8bit = image_8bit_contrast(nuclei_raw)

    nuclei_contours = nuclei_segment(nuclei_8bit, dest=path_out, threshold=150)

    image = cv2.cvtColor(nuclei_8bit, cv2.COLOR_GRAY2BGR)
    for n_id, cnt in enumerate(nuclei_contours):
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), thickness=7)
        cv2.putText(image, f'N{n_id}', org=cnt_centre(cnt), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=2, color=(255, 255, 255))

    # Detect foci
    centrioles_raw = channel_extract(projected, 1)
    centrioles_blur = cv2.GaussianBlur(centrioles_raw, (3, 3), sigmaX=0)
    centrioles_threshold = foci_detect(centrioles_blur, dest=path_out)
    masked = cv2.bitwise_and(centrioles_raw, centrioles_raw, mask=centrioles_threshold)
    masked = image_8bit_contrast(masked)
    masked = cv2.equalizeHist(masked)
    cv2.imwrite(str(path_out / 'clahe.png'), masked)
    centrioles_float = img_as_float(masked)
    foci_coords = np.fliplr(peak_local_max(centrioles_float, min_distance=5))

    image[:, :, 1] = image_8bit_contrast(masked)

    # Add the ground truth if available
    try:
        labels = labelbox_annotation_load('data/annotation.json', f'{dataset_name}_C1_000_000.png')
        for i, label in enumerate(labels):
            x, y = label_coordinates(label)
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 10, (200, 200, 200), 2)
    except IndexError:
        pass

    for f_id, (r, c) in enumerate(foci_coords):
        cv2.drawMarker(image, (r, c), (8, 8, 8), markerType=cv2.MARKER_CROSS, markerSize=5)

    # Segment centrosomes
    centrosomes_bboxes = centrosomes_box(centrioles_threshold)
    cv2.drawContours(image, centrosomes_bboxes, -1, (0, 255, 0))

    # Label the centrosomes
    centrosomes_mask = np.zeros((w, h), dtype=np.uint8)
    cv2.drawContours(centrosomes_mask, centrosomes_bboxes, -1, 255, -1)
    _, centrosomes_labels = cv2.connectedComponents(centrosomes_mask)
    labels_vis = 255 * (centrosomes_labels/centrosomes_labels.max())
    cv2.imwrite(str(path_out / 'centrosomes_labels.png'), labels_vis)

    # Label the nuclei
    nuclei_mask = np.zeros((w, h), dtype=np.uint8)
    cv2.drawContours(nuclei_mask, nuclei_contours, -1, 255, -1)
    _, nuclei_labels = cv2.connectedComponents(nuclei_mask)
    labels_vis = 255 * (nuclei_labels / nuclei_labels.max())
    cv2.imwrite(str(path_out / 'nuclei_labels.png'), labels_vis)

    cent2foci = []
    for f_id, (r, c) in enumerate(foci_coords):
        cent2foci.append((int(centrosomes_labels[r, c]), f_id))

    nuclei2cent = []
    for cent_id, cnt in enumerate(centrosomes_bboxes):
        r, c = cnt_centre(cnt)
        distances = [(n_id, cv2.pointPolygonTest(nucleus, (r, c), measureDist=True))
                     for n_id, nucleus in enumerate(nuclei_contours)]
        closest = max(distances, key=itemgetter(1))[0]
        nuclei2cent.append(closest)

    for c_id, cnt in enumerate(centrosomes_bboxes):
        r, c = cnt_centre(cnt)
        cv2.putText(image, f'C{c_id}', org=(r + 10, c), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5, thickness=1, color=(255, 255, 255))

    cv2.imwrite(str(path_out / f'{fov_name.split(".")[0]}_annot.png'), image)

    print(0)


if __name__ == '__main__':
    main()
