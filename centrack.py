import json
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

    blurred = cv2.medianBlur(nuclei, kernel_size)

    # Threshold the image
    threshold_mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    ret, nuclei_thresh = cv2.threshold(blurred, 0, 255, threshold_mode)

    return ret, nuclei_thresh


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


def main():
    dev_flag = False
    path_root = Path('/Volumes/work/datasets')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'
    fov_name = f'{dataset_name}_000_000.tif'

    path_fov = path_root / dataset_name / 'raw' / fov_name
    path_projected = path_root / dataset_name / 'projected' / fov_name

    projected = tf.imread(path_projected, key=range(4))

    nuclei_raw = channel_extract(projected, 0)
    nuclei = image_8bit_contrast(nuclei_raw)

    if dev_flag:
        cv2.imwrite('out/nuclei.png', nuclei)

    # Apply the gaussian blurring filter with large kernel (1/16 of the image width)
    image_w, image_h = nuclei.shape[-2:]
    kernel_size = image_w // 32 + 1

    nuclei_blurred = cv2.medianBlur(nuclei, kernel_size)
    nuclei_blurred = cv2.medianBlur(nuclei_blurred, kernel_size)
    if dev_flag:
        cv2.imwrite('out/nuclei_blurred.png', nuclei_blurred)

    # Threshold the image
    threshold_mode = cv2.THRESH_BINARY  # + cv2.THRESH_OTSU
    ret, nuclei_thresh = cv2.threshold(nuclei_blurred, 70, 255, threshold_mode)

    if dev_flag:
        cv2.imwrite('out/nuclei_thresh.png', nuclei_thresh)

    # Find the contours
    nuclei_contours, hierarchy = cv2.findContours(nuclei_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centrioles_raw = channel_extract(projected, 1)
    centrioles = cv2.medianBlur(centrioles_raw, 3)
    centrioles = cv2.GaussianBlur(centrioles, (3, 3), sigmaX=0)
    centrioles = image_8bit_contrast(centrioles)
    ret1, centrioles_thresh = cv2.threshold(centrioles, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if dev_flag:
        cv2.imwrite('out/centrioles.png', centrioles)
        cv2.imwrite('out/centrioles_thresh.png', centrioles_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    centrosomes = cv2.morphologyEx(centrioles_thresh, op=cv2.MORPH_DILATE, kernel=kernel, iterations=5)
    centrosomes_contours, hierarchy = cv2.findContours(centrosomes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if dev_flag:
        cv2.imwrite('out/centrosomes.png', centrosomes)

    masked = cv2.bitwise_and(centrioles_raw, centrioles_raw, mask=centrioles_thresh)

    centrioles_float = img_as_float(masked.copy())
    foci_coords = peak_local_max(centrioles_float, min_distance=2)

    image = cv2.cvtColor(nuclei_blurred, cv2.COLOR_GRAY2BGR)
    image[:, :, 1] = centrioles

    for n_id, cnt in enumerate(nuclei_contours):
        cv2.drawContours(image, [cnt], 0, (255, 0, 0))
        cv2.putText(image, f'N{n_id}', org=cnt_centre(cnt), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=2, color=(255, 255, 255))

    for f_id, (x, y) in enumerate(foci_coords):
        cv2.drawMarker(image, (y, x), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10)

    for c_id, cnt in enumerate(centrosomes_contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0))
        r, c = cnt_centre(cnt)
        cv2.putText(image, f'C{c_id}', org=(r + 10, c), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5, thickness=1, color=(255, 255, 255))

    # pdb.set_trace()
    labels = labelbox_annotation_load('data/annotation.json', 'RPE1wt_CEP63+CETN2+PCNT_1_C1_000_000.png')

    for i, label in enumerate(labels):
        x, y = label_coordinates(label)
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), 10, (200, 200, 200), 2)
        # cv2.putText(img=image, text=str(i), org=(x + 10, y + 10),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=1, color=(200, 200, 200))

    cv2.imwrite(f'out/{fov_name.split(".")[0]}_annot.png', image)
    print(0)

    cells_contours = cell_segment(nuclei)

    # results = []
    #
    # for n, cnt_cell in enumerate(cells_contours):
    #     for c, cnt_cm in enumerate(centrosomes_contours):
    #         centre = cnt_centre(cnt_cm)
    #         if cv2.pointPolygonTest(cnt_cell, centre, measureDist=False) < 0:
    #             for f, focus in enumerate(foci_coords):
    #                 if cv2.pointPolygonTest(cnt_cm, focus, measureDist=False) < 0:
    #                     results.append((n, c, f))


if __name__ == '__main__':
    main()
