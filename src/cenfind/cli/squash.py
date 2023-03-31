import argparse
import sys
from pathlib import Path

from cenfind.core.data import Dataset
from cenfind.core.log import get_logger


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser("squash", help="Write z-projections.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")

    return parser


def run(args):
    """
    Squash the raw fields of view.
    """

    dataset = Dataset(args.dataset)
    logger = get_logger(__name__, console=1, file=dataset.logs / 'cenfind.log')

    if dataset.has_projections:
        logger.info("Projections already exist, squashing skipped.")
        sys.exit()

    if not dataset.raw.exists():
        logger.error(
            "Folder raw/ does not exist. Make sure to move all your images under raw/"
        )
        raise FileNotFoundError

    dataset.write_projections()
    logger.info("Projections saved under %s" % dataset.projections)


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'))
    run(args)
