import sys
from pathlib import Path
from cenfind.core.data import Dataset

def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "prepare", help="Prepare the dataset structure"
    )
    parser.add_argument("path", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--projection_suffix",
        type=str,
        default="_max",
        help="Suffix indicating projection, e.g., `_max` (default) or `Projected`",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        help="Write the train and test splits for continuous learning using the channels specified",
    )

    return parser

def run(args):
    """Set up the dataset folder with default directories"""

    if not args.path.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)

    dataset.setup()
    dataset.write_fields()
    if args.splits:
        dataset.write_train_test(args.splits)
