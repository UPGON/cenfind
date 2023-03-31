import argparse
import sys
from pathlib import Path

from cenfind.core.data import Dataset
from cenfind.core.log import get_logger

logger = get_logger(__name__)


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "prepare", help="Prepare the dataset structure"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument("projection_suffix", type=str, help="the projection suffix like `_max`", default='_max')
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        help="Write the train and test splits for continuous learning using the channels specified",
    )

    return parser


def run(args):
    """Set up the dataset folder with default directories"""

    if not args.dataset.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    dataset = Dataset(args.dataset, projection_suffix=args.projection_suffix)
    logger.info("Dataset loaded")

    dataset.setup()
    logger.info("Dataset set up")
    dataset.write_fields()
    logger.info("Projections saved")

    if args.splits:
        dataset.write_splits(args.splits)


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              projection_suffix='_max',
                              splits=[1, 2])
    run(args)
