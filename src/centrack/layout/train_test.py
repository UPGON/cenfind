import random
from centrack.layout.dataset import DataSet
from centrack.utils.constants import PREFIX_LOCAL, datasets


def generate_channels(in_file, out_file):
    with open(in_file, 'r') as f:
        random.seed(1993)
        res = [f"{line.strip()},{random.randint(1, 3)}\n"
               for line in f.readlines()]

    with open(out_file, 'w') as out:
        for line in res:
            out.write(line)


def main():
    for ds in datasets:
        path_dataset = PREFIX_LOCAL / ds
        dataset = DataSet(path_dataset)
        train_images, test_images = dataset.splits(p=.9)
        with open(path_dataset / 'train.txt', 'w') as f:
            for fov in train_images:
                f.write(f"{fov}\n")

        with open(path_dataset / 'test.txt', 'w') as f:
            for fov in test_images:
                f.write(f"{fov}\n")

    for ds in datasets:
        generate_channels(PREFIX_LOCAL / ds / 'train.txt', PREFIX_LOCAL / ds / 'train_channels.txt')
        generate_channels(PREFIX_LOCAL / ds / 'test.txt', PREFIX_LOCAL / ds / 'test_channels.txt')


if __name__ == '__main__':
    main()
