import argparse
import logging
import sys
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        type=Path,
                        help='Path to the dataset')

    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` (default) or `Projected`')

    args = parser.parse_args()

    if not args.path.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    return args


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def main():
    args = get_args()
    path_dataset = args.path
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)

    dataset.projections.mkdir(exist_ok=True)
    logger.info('Projections folder created')

    dataset.write_fields()
    logger.info('Field registry written')

    dataset.predictions.mkdir(exist_ok=True)
    (dataset.predictions / 'centrioles').mkdir(exist_ok=True)
    (dataset.predictions / 'nuclei').mkdir(exist_ok=True)
    logger.info('Prediction folder created')

    dataset.statistics.mkdir(exist_ok=True)
    logger.info('Statistics folder created')

    dataset.visualisation.mkdir(exist_ok=True)
    logger.info('Visualisation folder created')

    dataset.vignettes.mkdir(exist_ok=True)
    logger.info('Vignettes folder created')


if __name__ == '__main__':
    main()
