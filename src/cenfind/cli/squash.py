import argparse
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tiff files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source',
                        type=Path,
                        help='Path to the dataset folder')
    return parser.parse_args()


def main():
    args = get_args()
    dataset = Dataset(args.source)
    dataset.write_projections()


if __name__ == '__main__':
    main()
