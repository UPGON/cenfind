import sys
from pathlib import Path
from cenfind.core.data import Dataset
import pytomlpp


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "prepare", help="Prepare the dataset structure"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--projection_suffix",
        type=str,
        default="",
        help="Suffix indicating projection, e.g., `_max` or `Projected`, empty if not specified",
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

    if not args.dataset.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    with open(args.dataset / "metadata.toml", "w") as f:
        content = pytomlpp.dumps({"projection_suffix": args.projection_suffix})
        f.write(content)

    dataset = Dataset(args.dataset, projection_suffix=args.projection_suffix)

    dataset.setup()
    dataset.write_fields()
    if args.splits:
        dataset.write_train_test(args.splits)
