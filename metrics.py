import numpy as np
from numpy.random import default_rng, SeedSequence
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import tifffile as tf
import cv2

rng = default_rng(1993)


def main():
    height, width = 2048, 2048
    object_number = 20
    noise_gaussian_scale = 1
    false_positives_n = int(.1 * object_number)
    false_negatives_n = int(.1 * object_number)

    # Generate ground truth objects (true positives)
    object_positions_actual = rng.integers(0, height, size=(object_number, 2))

    # Simulate the predictions and delete some objects (the false negatives)
    noise = rng.normal(scale=noise_gaussian_scale, size=(object_number, 2)).astype(int)
    object_positions_preds = object_positions_actual + noise
    false_negatives = rng.integers(0, object_number, false_negatives_n)
    object_positions_preds = np.delete(object_positions_preds, false_negatives, axis=0)

    # Add some false positives to the predictions
    false_positives = rng.integers(0, height, size=(false_positives_n, 2))
    object_positions_preds = np.concatenate([object_positions_preds, false_positives], axis=0)

    # Assign the predictions to the ground truth using the Hungarian algorithm.
    cost_matrix = cdist(object_positions_actual, object_positions_preds)
    row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=False)
    cost_overall = cost_matrix[row_inds, col_inds].sum()

    image = np.zeros((height, width), dtype=np.uint8)

    for focus in object_positions_actual:
        r, c = focus
        cv2.circle(image, radius=2, center=(r, c), color=255, thickness=-1)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    tf.imwrite('out/synthetic.png', image)


if __name__ == '__main__':
    main()
