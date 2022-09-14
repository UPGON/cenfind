import argparse
from pathlib import Path

import tifffile as tf
from tqdm import tqdm

from centrack.core.data import Dataset, Field


def main():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tiff files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source',
                        type=Path,
                        help='Path to the ds folder; the parent of `raw`.',
                        )
    args = parser.parse_args()

    path_dataset = args.source
    dataset = Dataset(path_dataset)

    for field in tqdm(dataset.pairs):
        field = Field(field, dataset)
        projection = field.stack.max(1)
        tf.imwrite(dataset.projections / f"{field.name}_max.tif", projection)


if __name__ == '__main__':
    main()
