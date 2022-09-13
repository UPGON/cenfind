import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from numpy.random import default_rng
import numpy as np
from spotipy.utils import points_matching

from centrack.data.base import Dataset, Field
from centrack.scoring.detectors import get_model
from centrack.scoring.detectors import detect_centrioles

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset',
                        type=Path,
                        help='Path to the dataset folder')
    parser.add_argument('model',
                        type=str,
                        help='Path to the model, e.g., <project>/models/dev/master')
    parser.add_argument('--offset',
                        type=int,
                        help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    dataset = Dataset(args.path_dataset)
    path_centrioles = args.path_dataset / 'annotations/centrioles'

    centriole_detector = get_model(args.model)

    performances = []

    fields_test = dataset.splits_for('test')
    fields_train = dataset.splits_for('train')
    fields_all = fields_train + fields_test

    for field, ch in fields_all:
        field = Field(field, dataset)
        annotation = field.annotation(channel_id=ch)

        predictions_np = detect_centrioles(field, channel=ch, model=centriole_detector)
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
    results = pd.DataFrame(performances)
    results.to_csv(args.path_dataset / f'precision_recall_{args.offset}.csv')
