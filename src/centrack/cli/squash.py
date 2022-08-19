import argparse
from pathlib import Path

from tqdm import tqdm

from centrack.data.base import Dataset, Stack


def cli():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tiff files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source',
                        type=Path,
                        help='Path to the ds folder; the parent of `raw`.',
                        )
    parser.add_argument('suffix',
                        type=str,
                        help='Format of the raw files, e.g., `.ome.tif` or `.stk`')

    args = parser.parse_args()

    path_dataset = args.source
    if not path_dataset.exists():
        raise FileNotFoundError(
            f'raw/ folder not found, please make sure to move the ome.tif files in {path_dataset}.')

    dataset = Dataset(path_dataset)

    if not dataset.projections.exists():
        dataset.projections.mkdir()

    files_raw = dataset.fields('raw')

    for path in tqdm(files_raw):
        stack = Stack(dataset, path.name)
        stack.write_projection()


if __name__ == '__main__':
    cli()
