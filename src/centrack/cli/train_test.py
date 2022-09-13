import argparse
from pathlib import Path

from centrack.core.data import Dataset


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

    train_pairs, test_pairs = dataset.split_train_test(args.channels, p=.9)

    with open(path_dataset / 'train.txt', 'w') as f:
        for fov, channel in train_pairs:
            f.write(f"{fov},{channel}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov, channel in test_pairs:
            f.write(f"{fov},{channel}\n")


if __name__ == '__main__':
    main()
