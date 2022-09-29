import argparse
import itertools
import random
from pathlib import Path
from typing import Any

from cenfind.core.data import Dataset


def choose_channel(fields: list[str], channels, unique=False) -> list[tuple[Any, Any]]:
    """Assign channel to field."""
    if len(fields) < len(channels):
        raise ValueError('not all channels will be represented in the set')
    if unique:
        return [(fov, channel)
                for fov, channel in zip(fields, itertools.cycle(channels))]
    else:
        return [(fov, channel) for fov, channel in itertools.product(fields, channels)]


def split_pairs(fields: list[tuple[str, int]], p=.9) -> tuple[Any, Any]:
    """
    Split a list of pairs (field, channel).

    :param fields
    :param p the train proportion, default to .9
    :return train_split, test_split
    """

    random.seed(1993)
    size = len(fields)
    split_idx = int(p * size)
    shuffled = random.sample(fields, k=size)
    split_test = shuffled[split_idx:]
    split_train = shuffled[:split_idx]

    return split_train, split_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    parser.add_argument('--unique', type=bool, default=False)
    parser.add_argument('--projection_suffix',
                        type=str,
                        default='max',
                        help='the suffix indicating projection, e.g., `max` or `Projected`')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    path_dataset = args.path
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)
    fields = dataset.fields
    train_fields, test_fields = split_pairs(fields, p=.9)
    pairs_train = choose_channel(train_fields, args.channels)
    pairs_test = choose_channel(test_fields, args.channels)

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov, channel in pairs_train:
            f.write(f"{fov},{channel}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov, channel in pairs_test:
            f.write(f"{fov},{channel}\n")


if __name__ == '__main__':
    main()
