import argparse
import sys
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tif files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset folder')
    return parser.parse_args()


def main():
    args = get_args()
    dataset = Dataset(args.path)

    if dataset.has_projections:
        print(f"Projections already exists, squashing aborted.")
        sys.exit()

    if not dataset.raw.exists():
        print(f"raw folder not found.")
        sys.exit()

    dataset.write_projections()


if __name__ == '__main__':
    main()
