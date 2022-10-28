import argparse
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` (default) or `Projected`')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    path_dataset = args.path
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)

    dataset.projections.mkdir(exist_ok=True)
    dataset.predictions.mkdir(exist_ok=True)
    dataset.visualisation.mkdir(exist_ok=True)
    dataset.statistics.mkdir(exist_ok=True)
    dataset.vignettes.mkdir(exist_ok=True)
    dataset.write_fields()
    dataset.write_train_test(args.channels)


if __name__ == '__main__':
    main()
