import cv2
import numpy as np

from centrack.annotation import Centre, Contour


def mask_from(annotation, w=2048, h=2048, radius=2):
    """Draw a mask of the objects (points, contours).
    This function computes quantitative masks not a visualisation
    """
    mask = np.zeros((w, h), dtype=np.uint8)

    for element in annotation:
        if isinstance(element, Centre):
            cv2.circle(mask, element.centre, radius, 1, thickness=-1)

        if isinstance(element, Contour):
            cv2.drawContours(mask, [element], 0, 1, thickness=-1)

    return mask


def overlap(mask_actual, mask_pred):
    """Make a visual estimation of the overlap of two masks."""
    if mask_actual.shape != mask_pred.shape:
        raise ValueError(f'mask_actual shape ({mask_actual.shape})!= mask_pred shape ({mask_pred.shape})')

    h, w = mask_actual.shape

    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    comparison[:, :, 0] = mask_pred
    comparison[:, :, 1] = mask_actual

    return comparison


def iou(mask_pred, mask_actual, w, h, radius):
    """Compute the intersection over union of two masks."""
    mask_and = np.logical_and(mask_actual, mask_pred)
    mask_or = np.logical_or(mask_actual, mask_pred)
    iou = ((mask_and.sum() + 1e-5) / (mask_or.sum() + 1e-5)).round(3)

    return iou
