import argparse
from pathlib import Path

from tqdm import tqdm

from centrack.data.base import Dataset, Stack


def main():
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
    dataset = Dataset(path_dataset)
    (dataset.path / 'projections').mkdir(exist_ok=True)

    files_raw = dataset.fields(args.suffix)

    for path in tqdm(files_raw):
        stack = Stack(dataset, path.name)
        stack.write_projection()


if __name__ == '__main__':
    main()
