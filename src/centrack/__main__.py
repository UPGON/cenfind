import argparse
import logging
from pathlib import Path

from centrack.commands.squash import cli as squash_cli
from centrack.commands.status import DataSet

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


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
    content = ds.check_projections()
    if not content:
        squash_cli()
    ds.check_predictions()


if __name__ == '__main__':
    cli()
