import argparse
from pathlib import Path
import random
from typing import List
from centrack.layout.dataset import DataSet


def generate_channels(in_file, out_file, channels: List[int]):
    with open(in_file, 'r') as f:
        random.seed(1993)
        res = [f"{line.strip()},{random.sample(channels, 1)[0]}\n"
               for line in f.readlines()]

    with open(out_file, 'w') as out:
        for line in res:
            out.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('suffix', type=str)
    parser.add_argument('channels', type=int, nargs='+')
    args = parser.parse_args()
    print(args.channels)
    path_dataset = args.path
    suffix = args.suffix
    dataset = DataSet(path_dataset)
    train_images, test_images = dataset.splits(suffix, p=.9)
    with open(path_dataset / 'train.txt', 'w') as f:
        for fov in train_images:
            f.write(f"{fov}\n")

    with open(path_dataset / 'test.txt', 'w') as f:
        for fov in test_images:
            f.write(f"{fov}\n")

    generate_channels(path_dataset / 'train.txt', path_dataset / 'train_channels.txt', channels=args.channels)
    generate_channels(path_dataset / 'test.txt', path_dataset / 'test_channels.txt', channels=args.channels)


if __name__ == '__main__':
    main()
