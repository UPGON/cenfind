import argparse
from pathlib import Path

import pandas as pd

from cenfind.core.data import Dataset
from cenfind.core.helpers import get_model
from cenfind.core.measure import dataset_metrics
from cenfind.experiments.constants import PREFIX_REMOTE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets',
                        type=Path,
                        nargs='+',
                        help='Path to the dataset folder, can be one or more')
    parser.add_argument('--model',
                        type=str,
                        help='Path to the model, e.g., <project>/models/dev/master')
    parser.add_argument('--tolerance',
                        type=int,
                        nargs='+',
                        help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    model = get_model(args.model)
    tolerance = args.tolerance
    datasets = args.datasets

    performances = []
    for dataset_name in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        performance = dataset_metrics(dataset, split=True, model=model, tolerance=tolerance)
        performances.append(performance)

    performances_df = pd.DataFrame([s for p in performances for s in p])

    path_out = Path('out')
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(path_out / f'performances_{args.model}.csv')


if __name__ == '__main__':
    main()
