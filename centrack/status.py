import argparse
import logging
from pathlib import Path

from describe import DataSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset')

    return parser.parse_args()


def main():
    logger.debug('Starting')
    args = parse_args()
    ds = DataSet(args.path)

    ds.check_description()
    ds.check_raw()
    ds.check_projections()
    ds.check_outline()

    ds.check_predictions()
    ds.check_annotations()


if __name__ == '__main__':
    main()
