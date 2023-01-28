import sys
from pathlib import Path
from cenfind.core.data import Dataset


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser("squash", help="Write z-projections.")
    parser.add_argument("path", type=Path, help="Path to the dataset folder")

    return parser


def run(args):
    """
    Squash the raw fields of view.
    """

    dataset = Dataset(args.path)

    if dataset.has_projections:
        print("Projections already exist, squashing skipped.")
        sys.exit()

    if not dataset.raw.exists():
        print(
            "ERROR: Folder raw/ does not exist. Make sure to move all your images under raw/"
        )
        sys.exit()

    dataset.write_projections()
    print("Projections saved under %s" % dataset.projections)
