import argparse
import sys
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset')

    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` (default) or `Projected`')

    args = parser.parse_args()

    if not args.path.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    return args


def main():
    args = get_args()
    path_dataset = args.path
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)

    dataset.setup()
    dataset.write_fields()


if __name__ == '__main__':
    main()
