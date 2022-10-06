import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from cenfind.core.data import Dataset
from cenfind.core.measure import dataset_metrics
from cenfind.experiments.constants import PREFIX_REMOTE
from cenfind.experiments.constants import datasets as std_ds
from tqdm import tqdm

tf.get_logger().setLevel('ERROR')
## GLOBAL SEED ##
tf.random.set_seed(3)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        type=Path,
                        help='Path to the model, e.g., <project>/models/dev/master')
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

    if args.datasets is None:
        datasets = std_ds
    else:
        datasets = args.datasets

    performances = []
    p_bar = tqdm(datasets)
    for dataset_name in p_bar:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        p_bar.set_description(dataset_name)
        for tol in tolerance:
            p_bar.set_postfix({"tolerance": tol})
            performance = dataset_metrics(dataset, split='test', model=args.model, tolerance=tol)
            performances.append(performance)

    performances_df = pd.DataFrame([s for p in performances for s in p])

    path_out = Path('out')
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(path_out / f'performances_{args.model.name}.csv')


if __name__ == '__main__':
    main()
