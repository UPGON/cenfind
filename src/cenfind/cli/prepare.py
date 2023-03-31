import argparse
from pathlib import Path

import pytomlpp

from cenfind.core.data import Dataset, choose_channel


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

    with open(args.dataset / "metadata.toml", "w") as f:
        content = pytomlpp.dumps({"projection_suffix": args.projection_suffix})
        f.write(content)

    dataset = Dataset(args.dataset, projection_suffix=args.projection_suffix)

    dataset.setup()
    dataset.write_fields()

    # TODO: Move the rest to training specific programs
    if args.splits:
        train_fields, test_fields = dataset.split_pairs(p=0.9)
        pairs_train = choose_channel(train_fields, args.splits)
        pairs_test = choose_channel(test_fields, args.splits)

        with open(dataset.path / "train.txt", "w") as f:
            for fov, channel in pairs_train:
                f.write(f"{fov.name},{channel}\n")

        with open(dataset.path / "test.txt", "w") as f:
            for fov, channel in pairs_test:
                f.write(f"{fov.name},{channel}\n")


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              projection_suffix='_max',
                              splits=[1, 2])
    run(args)
