import argparse
from pathlib import Path

from cenfind.core.data import Dataset
from cenfind.core.helpers import choose_channel, split_pairs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channels', type=int, nargs='+')
    parser.add_argument('--projection_suffix',
                        type=str,
                        default='_max',
                        help='the suffix indicating projection, e.g., `_max` (default) or `Projected`')

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
