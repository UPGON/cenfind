import random
from centrack.layout.constants import datasets, PREFIX


def generate_channels(in_file, out_file):
    with open(in_file, 'r') as f:
        random.seed(1993)
        res = [f"{line.strip()},{random.randint(1, 3)}\n"
               for line in f.readlines()]

    with open(out_file, 'w') as out:
        for line in res:
            out.write(line)


if __name__ == '__main__':
    for ds in datasets:
        generate_channels(PREFIX / ds / 'train.txt', PREFIX / ds / 'train_channels.txt')
        generate_channels(PREFIX / ds / 'test.txt', PREFIX / ds / 'test_channels.txt')
