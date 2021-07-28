import cv2
import numpy as np

from skimage import img_as_float
from skimage.feature import peak_local_max

from utils import (
    image_8bit_contrast,
    channel_extract,

)


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


def foci_process(image, ks, dist_foci, factor, blur_type=None):
    """
    Preproces the centriole marker and find local peaks
    :param image:
    :param ks:
    :param dist_foci:
    :param factor:
    :param blur_type:
    :return:
    """
    image_raw = image.copy()
    if blur_type == 'median':
        image = cv2.medianBlur(image, ks)
    if blur_type == 'gaussian':
        image = cv2.GaussianBlur(image, (ks, ks), sigmaX=0)

    # image = image_8bit_contrast(image)
    image_median = np.median(image)
    print(image_median)
    threshold = factor * image_median
    ret1, centrioles_threshold = cv2.threshold(image.astype(float), threshold, 255, cv2.THRESH_BINARY)
    centrioles_threshold = centrioles_threshold.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    centrioles_threshold = cv2.morphologyEx(centrioles_threshold, cv2.MORPH_ERODE, kernel)

    masked = cv2.bitwise_and(image_raw, image_raw, mask=centrioles_threshold)
    # masked = cv2.equalizeHist(masked)
    centrioles_float = img_as_float(masked)
    foci_coords = np.fliplr(peak_local_max(centrioles_float, min_distance=dist_foci))

    masked = image_8bit_contrast(masked)

    return masked, foci_coords


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
