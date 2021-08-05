import cv2
import numpy as np

from skimage import img_as_float
from skimage.feature import peak_local_max

from utils import (
    image_8bit_contrast,
)


def nuclei_segment(nuclei, dest=None, threshold=None):
    """
    Extract the nuclei into contours.
    :param nuclei:
    :param dest: if specified, write the results to the file
    :param threshold: if specified, use it instead of the derived.
    :return: the list of contours detected
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

    nuclei_contours, hierarchy = cv2.findContours(nuclei_thresh, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    if dest:
        cv2.imwrite(str(dest / 'nuclei_blurred.png'), nuclei_blurred)
        cv2.imwrite(str(dest / 'nuclei_thresh.png'), nuclei_thresh)

    return nuclei_contours


def foci_process(image, ks, dist_foci, factor, blur=None):
    """
    Preprocess the centriole marker and find local peaks
    :param image:
    :param ks:
    :param dist_foci:
    :param factor:
    :param blur:
    :return: Binary image of the foci and the list of their coordinates
    """
    image_raw = image.copy()
    if blur == 'median':
        image = cv2.medianBlur(image, ks)
    if blur == 'gaussian':
        image = cv2.GaussianBlur(image, (ks, ks), sigmaX=0)

    image_median = np.median(image)
    threshold = factor * image_median
    ret1, centrioles_threshold = cv2.threshold(image.astype(float), threshold, 255, cv2.THRESH_BINARY)
    centrioles_threshold = centrioles_threshold.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    centrioles_threshold = cv2.morphologyEx(centrioles_threshold, cv2.MORPH_ERODE, kernel)

    masked = cv2.bitwise_and(image_raw, image_raw, mask=centrioles_threshold)
    centrioles_float = img_as_float(masked)
    foci_coords = np.fliplr(peak_local_max(centrioles_float, min_distance=dist_foci))

    masked = image_8bit_contrast(masked)

    return masked, foci_coords


def centrosomes_box(centrioles_threshold, dest=None, iterations=5):
    """
    Derive the centrosome bounding boxes by dilating the foci.
    :param iterations: the number of iterations of dilations
    :param centrioles_threshold:
    :param dest: if specified, write the mask of dilated foci
    :return: List of box coordinates
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    centrosomes = cv2.morphologyEx(centrioles_threshold, op=cv2.MORPH_DILATE,
                                   kernel=kernel, iterations=iterations)
    centrosomes_contours, hierarchy = cv2.findContours(centrosomes, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)

    centrosomes_bboxes = []
    for c_id, cnt in enumerate(centrosomes_contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        centrosomes_bboxes.append(np.int0(box))
    if dest:
        cv2.imwrite(str(dest / 'centrosomes.png'), centrosomes)
    return centrosomes_bboxes
