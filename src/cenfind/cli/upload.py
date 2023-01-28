import argparse
from pathlib import Path

from dotenv import dotenv_values
from labelbox import Client

from cenfind.core.data import Dataset


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "upload", help="Upload the vignettes to labelbox"
    )
    parser.add_argument(
        "path", type=Path, help="Path to the dataset with existing vignettes"
    )
    parser.add_argument("--env", type=Path, help="Path to the .env file")

    return parser


def run(args):
    ds = Dataset(args.path)

    config = dotenv_values(args.env)
    client = Client(api_key=config["LABELBOX_API_KEY"])

    dataset = client.create_dataset(name=f"{ds.path.name}", iam_integration=None)

    asset = [
        {"row_data": str(path), "external_id": path.name}
        for path in sorted((ds.path / "vignettes").iterdir())
    ]
    dataset.create_data_rows(asset)
