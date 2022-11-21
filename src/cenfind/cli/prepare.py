import argparse
import logging
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path,
                        help='Path to the dataset')
    parser.add_argument('channels', type=int, nargs='+',
                        help='Channel indices to evaluate often ` 1 2 3`')
    parser.add_argument('--pixel_size', type=float,
                        default=.1025,
                        help='Pixel size in um, default `.1025`')
    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` (default) or `Projected`')

    args = parser.parse_args()

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
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix, pixel_size=args.pixel_size)

    dataset.projections.mkdir(exist_ok=True)
    logger.info('Projections folder created')

    dataset.write_fields()
    logger.info('Field registry written')

    if (dataset.path / 'test.txt').exists():
        answer = input('Do you want to overwrite the train/test.txt? [y/n]: ')
        while answer not in ['yes', 'y', 'no', 'n']:
            answer = input('Please give a meaning full answer.')
        if answer == 'y':
            dataset.write_train_test(args.channels)
            logger.info('Train/test files created')
        if answer == 'n':
            logger.info('Using existing train/test.txt')
    else:
        dataset.write_train_test(args.channels)
        logger.info('Train/test files created')

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
