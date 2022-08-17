import argparse
import logging
from pathlib import Path

import pandas as pd
from numpy.random import default_rng

from centrack.data.base import Dataset, Projection, get_model, Channel
from centrack.data.measure import evaluate_one_image

logging.basicConfig(level=logging.INFO)

rng = default_rng(1993)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset', type=Path, help='Path to the ds folder')
    parser.add_argument('model', type=str, help='Path to the model, e.g., <project>/models/dev/master')
    parser.add_argument('--offset', type=int, help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    dataset = Dataset(args.path_dataset)
    path_centrioles = args.path_dataset / 'annotations/centrioles'

    centriole_detector = get_model('/home/buergy/projects/centrack/models/dev/2022-08-10_18:09:45')

    performances = []

    projs_test = dataset.splits_for('test')
    for proj_name, ch in projs_test:
        proj = Projection(dataset, proj_name)
        channel = Channel(proj, ch)
        annotation = channel.annotation()
        results = evaluate_one_image(channel, centriole_detector,
                                     annotation, offset_max=args.offset)
        performances.append(results)
        logging.info('Image: %s; Channel: %s; Precision : %.3f; Recall : %.3f',
                     proj.name, ch, results.precision, results.recall)
    results = pd.DataFrame(performances)
    results.to_csv(args.path_dataset / f'precision_recall_{args.offset}.csv')
