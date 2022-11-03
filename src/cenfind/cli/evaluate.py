import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.measure import dataset_metrics
from cenfind.experiments.constants import PREFIX_REMOTE
from cenfind.experiments.constants import datasets as std_ds

tf.get_logger().setLevel('ERROR')
## GLOBAL SEED ##
tf.random.set_seed(3)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        type=Path,
                        help='Path to the model, e.g., <project>/models/master')
    parser.add_argument('--datasets',
                        type=Path,
                        nargs='+',
                        help='Path to the dataset folder, can be one or more')
    parser.add_argument('--tolerance',
                        type=int,
                        nargs='+',
                        help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    tolerance = args.tolerance

    if len(tolerance) > 1:
        tolerance = np.linspace(tolerance[0], tolerance[1], 10)

    if args.datasets is None:
        datasets = std_ds
    else:
        datasets = args.datasets
    th_range = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    # th_range = [round(i, 3) for i in np.random.uniform(size=50)]
    performances = []
    p_bar = tqdm(datasets)
    for dataset_name in p_bar:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        p_bar.set_description(dataset_name)
        for tol in tolerance:
            for th in th_range:
                p_bar.set_postfix({"tolerance": tol, "threshold": th})
                prob_maps, performance = dataset_metrics(dataset, split='test', model=args.model, tolerance=tol,
                                                         threshold=th)
                performances.append(performance)

    performances_df = pd.DataFrame([s for p in performances for s in p])

    path_out = Path('out')
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(path_out / f'performances_{args.model.name}.csv')


if __name__ == '__main__':
    main()
