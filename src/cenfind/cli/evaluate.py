from numpy.random import seed

seed(1)

import tensorflow as tf

tf.random.set_seed(2)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.measure import dataset_metrics

tf.get_logger().setLevel('ERROR')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets',
                        type=Path,
                        nargs='+',
                        help='Path to the dataset folder, can be one or more')
    parser.add_argument('model',
                        type=Path,
                        help='Path to the model, e.g., <project>/models/master_20221123')
    parser.add_argument('--performances_file',
                        type=Path,
                        help='Path to the destination file for performances')
    parser.add_argument('--tolerance',
                        type=int,
                        nargs='+',
                        help='Distance above which two points are deemed not matching')
    parser.add_argument('--thresholds',
                        action='store_true',
                        help='Probability cutoff, if set, evaluate over [0, .95]')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = [2]
    _thresholds = args.thresholds
    if args.performances_file is None:
        dst = Path('./performances.csv')
    else:
        dst = Path(args.performances_file)
    while dst.exists():
        answer = input(f'Do you want to overwrite {dst}? [yn]: ')
        if answer == 'y':
            break
        if answer == 'n':
            dst = input('Please enter the destination path: ')
            dst = Path(dst)

    if len(tolerance) == 2:
        lower, upper = tolerance
        tolerance = list(range(lower, upper + 1))

    if _thresholds:
        thresholds = np.linspace(0, 1, 10, endpoint=False).round(1)
        thresholds = np.append(thresholds, .95)
    else:
        thresholds = [.5]

    datasets = args.datasets

    print(tolerance)

    datasets = [Dataset(d) for d in datasets]

    performances = []
    p_bar = tqdm(datasets)
    for dataset in p_bar:
        p_bar.set_description(dataset.path.name)
        for tol in tolerance:
            for th in thresholds:
                prob_maps, performance = dataset_metrics(dataset,
                                                         split='test',
                                                         model=args.model,
                                                         tolerance=tol,
                                                         threshold=th)
                performances.append(performance)
                p_bar.set_postfix({"tolerance": tol,
                                   "threshold": th,
                                   "performance peek": performance[0]['f1']})

    performances_df = pd.DataFrame([s for p in performances for s in p])
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(dst)


if __name__ == '__main__':
    main()
