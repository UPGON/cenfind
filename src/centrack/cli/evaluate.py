import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from numpy.random import default_rng
import numpy as np
from spotipy.utils import points_matching

from centrack.data.base import Dataset, Projection, Field, Channel
from centrack.data.base import get_model

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset', type=Path, help='Path to the dataset folder')
    parser.add_argument('model', type=str, help='Path to the model, e.g., <project>/models/dev/master')
    parser.add_argument('--offset', type=int, help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    dataset = Dataset(args.path_dataset)
    path_centrioles = args.path_dataset / 'annotations/centrioles'

    centriole_detector = get_model(args.model)

    performances = []

    projs_test = dataset.splits_for('test')
    projs_train = dataset.splits_for('train')
    projs_all = projs_train + projs_test
    for proj_name, ch in projs_all:
        field = Field(proj_name)
        proj = Projection(dataset, field)
        channel = Channel(proj, ch)
        annotation = channel.annotation()
        predictions = channel.detect_centrioles(model=centriole_detector)
        predictions_np = np.asarray([c.to_numpy() for c in predictions])
        annotation_np = np.asarray([a for a in annotation])
        if all((len(predictions_np), len(annotation_np))) > 0:
            results = points_matching(annotation_np, predictions_np, cutoff_distance=args.offset)
        else:
            logging.warning('detected: %d; annotated: %d... Set precision and accuracy to zero' % (
            len(predictions_np), len(predictions_np)))
            results = SimpleNamespace()
            results.precision = 0
            results.recall = 0

        performances.append(results)
        logging.info('Image: %s; Channel: %s; Precision : %.3f; Recall : %.3f',
                     proj.field.name, ch, results.precision, results.recall)
    results = pd.DataFrame(performances)
    results.to_csv(args.path_dataset / f'precision_recall_{args.offset}.csv')
