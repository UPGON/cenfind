import argparse
from pathlib import Path

from dotenv import dotenv_values
from labelbox import Client

from cenfind.core.data import Dataset


def main():
    parser = argparse.ArgumentParser(description="Upload the vignettes to labelbox")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    config = dotenv_values(".env")

    client = Client(api_key=config["LABELBOX_API_KEY"])
    ds = Dataset(args.path)

    dataset = client.create_dataset(name=f"{ds.path.name}", iam_integration=None)

    asset = [
        {"row_data": str(path), "external_id": path.name}
        for path in sorted((ds.path / "vignettes").iterdir())
    ]
    dataset.create_data_rows(asset)


if __name__ == "__main__":
    main()
