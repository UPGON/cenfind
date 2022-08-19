import argparse
import itertools
import logging
import sys
from pathlib import Path

from centrack.data.base import Dataset
from centrack.data.base import split_train_test

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    args = parser.parse_args()

    path_dataset = args.path
    dataset = Dataset(path_dataset)

    logging.info("%s" % dataset.file_name)

    if not dataset.projections:
        sys.exit()

    fovs = [f.name for f in dataset.fields('raw')]
    train_split, test_split = split_train_test(fovs, p=.9)

    train_pairs = [(fov, channel)
                   for fov, channel in zip(train_split, itertools.cycle(args.channels))]
    test_pairs = [(fov, channel)
                  for fov, channel in zip(test_split, itertools.cycle(args.channels))]

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov, channel in train_pairs:
            f.write(f"{fov},{channel}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov, channel in test_pairs:
            f.write(f"{fov},{channel}\n")


if __name__ == '__main__':
    main()
