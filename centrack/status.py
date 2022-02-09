import argparse
import logging
from pathlib import Path

from centrack.describe import DataSet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATTERNS = {
    'hatzopoulos': r'([\w\d]+)_(?:([\w\d-]+)_)?([\w\d\+]+)_(\d)',
    'garcia': r'^(?:\d{8})_([\w\d-]+)_([\w\d_-]+)_([\w\d\+]+)_((?:R\d_)?\d+)?_MMStack_Default'
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset')

    return parser.parse_args()


def cli():
    logger.debug('Starting')
    args = parse_args()
    ds = DataSet(args.path)

    ds.check_conditions()
    ds.check_raw()
    ds.check_projections()
    ds.check_outline()

    # ds.check_predictions()
    # ds.check_annotations()


if __name__ == '__main__':
    cli()
