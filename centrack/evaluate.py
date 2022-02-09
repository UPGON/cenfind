import logging

import numpy as np
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from centrack.describe import Condition
from centrack.fetch import fetch_files, Field
from centrack.outline import Centre
from centrack.score import extract_centrioles

logging.basicConfig(level=logging.INFO)

CYAN_RGB = (0, 255, 255)
RED_RGB = (255, 0, 0)
WHITE_RGB = (255, 255, 255)
rng = default_rng(1993)


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


def compute_metrics(positions, predictions, offset_max=2):
    """
    Assign the predictions to the ground truth using the Hungarian algorithm.
    :param positions: List of Points, annotations
    :param predictions: List of Points, predictions
    :param offset_max: number of pixels, above which two positions are
    considered different.
    :return: a dictionary of the fps, fns, and tps
    """
    cost_matrix = cdist(positions, predictions)
    agents, tasks = linear_sum_assignment(cost_matrix, maximize=False)

    tps = set()
    fps = set()
    fns = set()

    for agent, task in zip(agents, tasks):
        if cost_matrix[agent, task] > offset_max:
            fps.add(task)
            fns.add(agent)
        else:
            tps.add(task)

    pos_idx = set(range(len(positions)))
    new_fns = pos_idx.difference(set(agents))
    fns = fns.union(new_fns)

    pred_idx = set(range(len(predictions)))
    new_fps = pred_idx.difference(set(tasks))
    fps = fps.union(new_fps)

    return {
        'fp': len(fps),
        'fn': len(fns),
        'tp': len(tps)
        }


def compute_precision(tp: int, fp: int):
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int):
    return tp / (tp + fn)


def precision_recall(metrics):
    """
    Compute the precision and recall using the metrics.
    :param metrics: Dict containing the numbers of fp, fn, tp.
    :return: Dict with precision and recall
    """
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']

    return {'precision': compute_precision(tp, fp),
            'recall': compute_recall(tp, fn)}


def process_one_image(data, annotation):
    predictions = extract_centrioles(data)
    predictions_np = np.asarray([c.to_numpy() for c in predictions])
    positions_np = predictions_np
    confusion_matrix = compute_metrics(positions_np, predictions_np,
                                       offset_max=2)

    return precision_recall(confusion_matrix)


def cli():
    images = fetch_files(
        '/Volumes/work/epfl/datasets/20210727_HA-FL-SAS6_Clones/projections',
        file_type='.tif')
    condition = Condition(markers='DAPI+rPOC5AF488+mHA568+gCPAP647'.split('+'))

    for image in images[:2]:
        image = Field(image, condition).load()
        data = image[1, :, :].to_numpy()
        positions = np.asarray([Centre((500, 500)).to_numpy()])
        results = process_one_image(data, positions)
        logging.info('Image: %s: Precision : %.3f; Recall : %.3f',
                     image.name,
                     results['precision'],
                     results['recall'])


if __name__ == "__main__":
    cli()
