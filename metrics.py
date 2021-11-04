import logging

import cv2
import numpy as np
import tifffile as tf
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, euclidean

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)

CYAN_RGB = (0, 255, 255)
RED_RGB = (255, 0, 0)
WHITE_RGB = (255, 255, 255)


def main():
    height, width = 512, 512
    object_number = 50
    preds_random = False
    has_daughter = .8
    false_positives_rate = 0.1
    false_negative_rate = 0.8
    false_positives_n = int(false_positives_rate * object_number)
    false_negatives_n = int(false_negative_rate * object_number)

    # Generate ground truth objects (true positives)
    object_positions_actual = rng.integers(0, height, size=(object_number, 2))
    procentrioles = rng.choice(object_positions_actual, int(has_daughter * object_number),
                               replace=False) + rng.integers(-4, 4, size=(int(has_daughter * object_number), 2))
    object_positions_actual = np.concatenate([object_positions_actual, procentrioles])

    object_number, _ = object_positions_actual.shape

    # Simulate the predictions and delete some objects (the false negatives)
    object_positions_preds = object_positions_actual.copy()
    object_positions_preds = rng.choice(object_positions_preds, object_number - false_negatives_n, replace=False)

    # Add some false positives to the predictions
    if preds_random:
        object_positions_preds = rng.integers(0, height, size=(50, 2))
    else:
        false_positives = rng.integers(0, height, size=(false_positives_n, 2))
        object_positions_preds = np.concatenate([object_positions_preds, false_positives], axis=0)

    # Assign the predictions to the ground truth using the Hungarian algorithm.
    cost_matrix = cdist(object_positions_actual, object_positions_preds)
    agents, tasks = linear_sum_assignment(cost_matrix, maximize=False)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate the ground truth image
    for focus in object_positions_actual:
        r, c = focus
        cv2.circle(image, radius=2, center=(r, c), color=WHITE_RGB, thickness=-1)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    annotation = np.zeros_like(image)

    # Draw the matched predictions
    false_negatives = []
    true_positives = []
    false_positives = []
    for agent, task in zip(agents, tasks):
        actual = object_positions_actual[agent]
        pred = object_positions_preds[task]

        distance = euclidean(actual, pred)

        logging.info('distance %s', distance)

        if distance < 2:
            true_positives.append(agent)
        else:
            false_positives.append(task)

    # false_positives = set(range(len(object_positions_preds))).difference(set(tasks))

    for idx in false_positives:
        r, c = object_positions_preds[idx]
        cv2.drawMarker(annotation, position=(r, c), color=RED_RGB, thickness=1,
                       markerSize=10, markerType=cv2.MARKER_TILTED_CROSS)

    for idx in true_positives:
        r, c = object_positions_actual[idx]
        cv2.drawMarker(annotation, position=(r, c), color=CYAN_RGB, thickness=1,
                       markerSize=10, markerType=cv2.MARKER_SQUARE)

    for idx in false_negatives:
        r, c = object_positions_actual[idx]
        cv2.drawMarker(annotation, position=(r, c), color=RED_RGB, thickness=1,
                       markerSize=10, markerType=cv2.MARKER_CROSS)

    # Write the image
    tf.imwrite('out/synthetic.png', image + annotation)


if __name__ == '__main__':
    main()
