import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from numpy.random import default_rng

import tifffile as tf

from centrack.utils.status import DataSet
from centrack.commands.score import extract_centrioles, get_model
from spotipy.spotipy.utils import points_matching

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)


def process_one_image(data, model, annotation, offset_max=2, predictions=None):
    if predictions is None:
        predictions = extract_centrioles(data, model=model)
    predictions_np = np.asarray([c.to_numpy() for c in predictions])
    annotation_np = np.asarray([a for a in annotation])
    return points_matching(annotation_np, predictions_np,
                           cutoff_distance=offset_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset', type=Path, help='Path to the dataset folder')
    args = parser.parse_args()

    dataset = DataSet(args.path_dataset)
    path_centrioles = args.path_dataset / 'annotations/centrioles'

    centriole_detector = get_model(
        # '/Users/buergy/Dropbox/epfl/projects/centrack/models/test_model/2022-07-14-15-54-59.555651'
        '/Users/buergy/Dropbox/epfl/projects/centrack/models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300'
    )

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
            performances.append(results)
            logging.info('Image: %s; Channel: %s; Precision : %.3f; Recall : %.3f',
                         fov.name,
                         ch,
                         results['precision'],
                         results['recall'])
    results = pd.DataFrame(performances)
    results.to_csv(args.path_dataset / 'precision_recall.csv')
