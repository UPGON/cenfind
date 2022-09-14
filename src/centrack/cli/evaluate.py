import argparse
from pathlib import Path

import pandas as pd

from centrack.core.data import Dataset
from centrack.experiments.constants import PREFIX_REMOTE
from centrack.experiments.compare_detectors import get_model
from centrack.core.measure import run_evaluation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets',
                        type=Path,
                        nargs='+',
                        help='Path to the dataset folder, can be one or more')
    parser.add_argument('--model',
                        type=str,
                        help='Path to the model, e.g., <project>/models/dev/master')
    parser.add_argument('--tolerances',
                        type=int,
                        nargs='+',
                        help='Distance above which two points are deemed not matching')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    model = get_model(args.model)
    tolerances = list(range(6)) if args.tolerances is None else args.tolerances
    datasets = args.datasets

    performances = []
    for dataset_name in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        performance = run_evaluation(dataset, test_only=True, model=model,
                                     tolerances=tolerances)
        performances.append(performance)

    performances_df = pd.DataFrame([s for p in performances for s in p])

    path_out = Path('out')
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(path_out / f'performances_{args.model}.csv')


if __name__ == '__main__':
    main()
