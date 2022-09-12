import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Tuple, List

from centrack.data.base import Dataset


def split_train_test(items, p=.9) -> Tuple[List, List]:
    """
    Assign the FOV between train and test
    :param items:
    :param p: the fraction of train examples, by default .9
    :return: a tuple of lists
    """
    random.seed(1993)

    size = len(items)
    split_idx = int(p * size)
    shuffled = random.sample(items, k=size)
    split_test = shuffled[split_idx:]
    split_train = shuffled[:split_idx]

    return split_train, split_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    args = parser.parse_args()

    path_dataset = args.path
    dataset = Dataset(path_dataset)

    if not dataset.projections:
        sys.exit()

    fovs = dataset.fields()
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
