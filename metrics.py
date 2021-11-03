import numpy as np
from numpy.random import default_rng, SeedSequence
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import linear_sum_assignment
import tifffile as tf
import cv2

rng = default_rng(1993)


def main():
    height, width = 2048, 2048
    object_number = 50
    has_daughter = .8
    false_rate = 0
    false_positives_n = int(false_rate * object_number)
    false_negatives_n = int(false_rate * object_number)

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
    false_positives = rng.integers(0, height, size=(false_positives_n, 2))
    object_positions_preds = np.concatenate([object_positions_preds, false_positives], axis=0)
    # object_positions_preds = object_positions_preds + rng.normal(scale=10, size=(object_number, 2)).astype(int)

    # Assign the predictions to the ground truth using the Hungarian algorithm.
    cost_matrix = cdist(object_positions_actual, object_positions_preds)
    mapping = linear_sum_assignment(cost_matrix, maximize=False)
    agents, tasks = mapping
    cost_overall = cost_matrix[agents, tasks].sum()

    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate the ground truth image
    for focus in object_positions_actual:
        r, c = focus
        cv2.circle(image, radius=2, center=(r, c), color=(255, 255, 255), thickness=-1)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Draw the matched predictions
    for pred in object_positions_preds:
        cv2.drawMarker(image, position=pred, color=(255, 255, 255), markerSize=10,
                       markerType=cv2.MARKER_SQUARE)

    # Draw the matched predictions
    error = 0
    for agents_ind, tasks_ind in zip(agents, tasks):
        object_actual = object_positions_actual[agents_ind]
        object_pred = object_positions_preds[tasks_ind]

        distance = euclidean(object_pred, object_actual)

        print(f"{agents_ind:<4} -> {tasks_ind:>4}: dist = {int(distance):>5}")
        if distance < 5:
            cv2.arrowedLine(image, object_positions_preds[tasks_ind], object_positions_actual[agents_ind],
                            color=(0, 255, 0),
                            thickness=2,
                            tipLength=.1)
        else:
            error += 1
    print(error)

    # Write the image
    tf.imwrite('out/synthetic.png', image)


if __name__ == '__main__':
    main()
