import argparse
import itertools
import random
from pathlib import Path
from typing import List, Tuple

from centrack.core.data import Dataset


def write_fields(dataset):
    """
    Write field names to fields.txt.
    """
    if (dataset.path / 'raw').exists():
        folder = dataset.path / 'raw'
    elif (dataset.path / 'projections').exists():
        folder = dataset.path / 'projections'
    else:
        raise FileNotFoundError(dataset.path)

    fields = []
    for f in folder.iterdir():
        if f.name.startswith('.'):
            continue

        fields.append(f.name.split('.')[0].rstrip('_max'))

    with open(dataset.path / 'fields.txt', 'w') as f:
        for field in fields:
            f.write(field + '\n')


def split_train_test(dataset, channels: List[int], p=.9) -> Tuple[List, List]:
    """
    Assign the FOV between train and test
    :param channels:
    :param p: the fraction of train examples, by default .9
    :return: a tuple of lists
    """
    random.seed(1993)
    items = dataset.fields()
    size = len(items)
    split_idx = int(p * size)
    shuffled = random.sample(items, k=size)
    split_test = shuffled[split_idx:]
    split_train = shuffled[:split_idx]

    train_pairs = [(fov, channel)
                   for fov, channel in zip(split_train, itertools.cycle(channels))]
    test_pairs = [(fov, channel)
                  for fov, channel in zip(split_test, itertools.cycle(channels))]
    return train_pairs, test_pairs


# with open(self.path / 'fields.txt', 'r') as f:
#     fields = [line.rstrip() for line in f.readlines()]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    path_dataset = args.path
    dataset = Dataset(path_dataset)

    train_pairs, test_pairs = split_train_test(dataset, args.channels, p=.9)

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov, channel in train_pairs:
            f.write(f"{fov},{channel}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov, channel in test_pairs:
            f.write(f"{fov},{channel}\n")


if __name__ == '__main__':
    main()
