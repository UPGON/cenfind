import argparse
import logging
from pathlib import Path

from cenfind.core.data import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path,
                        help='Path to the dataset')
    parser.add_argument('channels', type=int, nargs='+',
                        help='Channel indices of centriolar markers to evaluate; often `1 2 3`.')

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
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)

    if (dataset.path / 'test.txt').exists():
        answer = input('Do you want to overwrite the train/test.txt? [y/n]: ')
        while answer not in ['y', 'n']:
            answer = input('Please enter [y/n].')
        if answer == 'y':
            dataset.write_train_test(args.channels)
            logger.info('Train/test files created')
        if answer == 'n':
            logger.info('Using existing train/test.txt')
    else:
        dataset.write_train_test(args.channels)
        logger.info('Train/test files created')

if __name__ == '__main__':
    main()
