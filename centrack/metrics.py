import cv2
import numpy as np
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, euclidean

from centrack.annotation import Centre, Contour

rng = default_rng(1993)


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


def generate_synthetic_data(height=512, size=200, has_daughter=.8):
    # Generate ground truth objects (true positives)
    foci = rng.integers(0, height, size=(size, 2))
    daughter_n = int(has_daughter * size)
    offset = rng.integers(-4, 4, size=(daughter_n, 2))

    daughters = rng.choice(foci, daughter_n, replace=False) + offset
    foci = np.concatenate([foci, daughters])

    return foci


def generate_predictions(height, foci, fn_rate=.1, fp_rate=.2, random=False):
    size = len(foci)

    fp_n = int(fp_rate * size)
    fn_n = int(fn_rate * size)

    if random:
        predictions = rng.integers(0, height, size=(50, 2))
        return predictions
    else:
        # Simulate the predictions and delete some objects (the false negatives)
        predictions = foci.copy()
        predictions = rng.choice(predictions, size - fn_n, replace=False)
        fps = rng.integers(0, height, size=(fp_n, 2))
        predictions = np.concatenate([predictions, fps], axis=0)

        return predictions


def compute_metrics(positions, predictions, offset_max):
    # Assign the predictions to the ground truth using the Hungarian algorithm.
    cost_matrix = cdist(positions, predictions)
    agents, tasks = linear_sum_assignment(cost_matrix, maximize=False)

    # Draw the matched predictions
    fns = []
    tps = []

    for agent, task in zip(agents, tasks):
        actual = positions[agent]
        pred = predictions[task]

        distance = euclidean(actual, pred)

        logging.info('distance %i', distance)

        if distance < offset_max:
            tps.append(agent)
        else:
            fns.append(agent)

    fps = set(range(len(predictions))).difference(set(tasks))

    return {'fp': fps,
            'fn': fns,
            'tp': tps}
