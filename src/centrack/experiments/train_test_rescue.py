from centrack.data.base import Dataset, split_train_test
from centrack.experiments.constants import PREFIX_REMOTE, datasets
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s")


def main():
    for dataset_name in datasets:
        path_dataset = PREFIX_REMOTE / dataset_name
        dataset = Dataset(path_dataset)
        logging.info("Processing %s" % dataset_name)

        pairs = []
        for path in (dataset.annotations / 'centrioles').iterdir():
            fov_name, channel_id = path.stem.split('_max_C')
            pairs.append((fov_name, int(channel_id)))

        channels_to_sample = [1, 2, 3]
        test_pairs = []
        while channels_to_sample:
            channel = channels_to_sample.pop()
            for p in pairs:
                if p[1] == channel:
                    test_pairs.append(p)
                    pairs.remove(p)
                    break
        train_pairs = pairs

        with open(path_dataset / 'train.txt', 'w') as f:
            for fov, channel in train_pairs:
                f.write(f"{fov},{channel}\n")
        logging.info("Writing train.txt")

        with open(path_dataset / 'test.txt', 'w') as f:
            for fov, channel in test_pairs:
                f.write(f"{fov},{channel}\n")
        logging.info("Writing test.txt")


if __name__ == '__main__':
    main()
