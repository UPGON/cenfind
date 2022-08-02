import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import tifffile as tf

from centrack.utils.status import DataSet
from centrack.utils.outline import Centre
from centrack.commands.score import extract_centrioles, get_model

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)


def generate_synthetic_data(height=512, size=200, has_daughter=.8):
    """
    Generate ground truth objects (true positives)
    :param height:
    :param size:
    :param has_daughter:
    :return:
    """
    _foci = rng.integers(0, height, size=(size, 2))
    daughter_n = int(has_daughter * size)
    offset = rng.integers(-4, 4, size=(daughter_n, 2))

    daughters = rng.choice(_foci, daughter_n, replace=False) + offset
    _foci = np.concatenate([_foci, daughters], axis=0)

    return [Centre(f) for f in _foci]


def generate_predictions(height: int, foci, fn_rate=.1, fp_rate=.2):
    """
    Generate predictions from annotation
    :param height:
    :param foci:
    :param fn_rate:
    :param fp_rate:
    :return: List[Centre]
    """
    size = len(foci)

    fp_n = int(fp_rate * size)
    fn_n = int(fn_rate * size)

    predictions = foci.copy()
    predictions = rng.choice(predictions, size - fn_n, replace=False)
    fps = rng.integers(0, height, size=(fp_n, 2))
    predictions = np.concatenate([predictions, fps], axis=0)

    return [Centre(f) for f in predictions]


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
    return round(tp / (tp + fp), 3)


def compute_recall(tp: int, fn: int):
    return round(tp / (tp + fn), 3)


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


def process_one_image(data, model, annotation, offset_max=2, predictions=None):
    if predictions is None:
        predictions = extract_centrioles(data, model=model)
    predictions_np = np.asarray([c.to_numpy() for c in predictions])
    annotation_np = np.asarray([a for a in annotation])
    confusion_matrix = compute_metrics(annotation_np, predictions_np,
                                       offset_max=offset_max)

    return precision_recall(confusion_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset', type=Path, help='Path to the dataset folder')
    args = parser.parse_args()

    dataset = DataSet(args.path_dataset)
    path_centrioles = args.path_dataset / 'annotations/centrioles'

    centriole_detector = get_model(
        '/home/buergy/projects/centrack/models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300')

    performances = []

    for fov in dataset.projections.iterdir():
        data = tf.imread(fov)
        for ch in range(1, data.shape[0]):
            channel = data[ch, :, :]
            annotation_file = fov.name.replace('.tif', f'_C{ch}.txt')
            foci_path = str(path_centrioles / annotation_file)
            try:
                foci = np.loadtxt(foci_path, dtype=int, delimiter=',')
            except FileNotFoundError:
                logging.info('Annotation file not found (%s)', str(foci_path))
                continue
            results = process_one_image(channel, centriole_detector, foci, offset_max=2)
            results['fov'] = fov.name
            results['channel'] = ch
            performances.append(results)
            logging.info('Image: %s; Channel: %s; Precision : %.3f; Recall : %.3f',
                         fov.name,
                         ch,
                         results['precision'],
                         results['recall'])
    results = pd.DataFrame(performances)
    results.to_csv(args.path_dataset / 'precision_recall.csv')
