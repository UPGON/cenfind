from pathlib import Path
import json
import numpy as np
import tifffile as tif
import tensorflow as tf

from spotipy.spotipy.utils import points_to_prob
from spotipy.spotipy.model import SpotNet, Config
from spotipy.spotipy.utils import normalize_fast2d

from centrack.layout.dataset import DataSet
from centrack.inference.score import get_model
from centrack.layout.constants import datasets


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, 'r') as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def load_pairs(dataset: DataSet, split='train'):
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    images = []
    positions = []

    fovs = dataset.split_images_channel(split)

    for fov, channel_id in fovs:
        channel_id = int(channel_id)
        image_path = str(dataset.projections / f"{fov}_max.tif")
        image = tif.imread(image_path)
        image = image[channel_id, :, :]
        # image_norm = image.astype('float32') / image.max()
        image_norm = normalize_fast2d(image, clip=True)
        images.append(image_norm)

        foci_path = str(dataset.annotations / 'centrioles' / f"{fov}_max_C{channel_id}.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')  # in format x, y; origin at top left

        foci_mask = points_to_prob(foci, shape=image.shape, sigma=1)
        positions.append(foci_mask)

    return np.stack(images), np.stack(positions)


def main():
    all_train_x = []
    all_train_y = []
    all_test_x = []
    all_test_y = []

    for dataset_name in datasets:
        dataset = DataSet(Path(f'/Users/buergy/Dropbox/epfl/datasets/{dataset_name}'))
        train_x, train_y = load_pairs(dataset, split='train')
        test_x, test_y = load_pairs(dataset, split='test')
        all_train_x.append(train_x)
        all_train_y.append(train_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)

    all_train_x = np.concatenate(all_train_x)
    all_train_y = np.concatenate(all_train_y)
    all_test_x = np.concatenate(all_test_x)
    all_test_y = np.concatenate(all_test_y)

    config = read_config('models/dev/config.json')
    model = SpotNet(config, name='all_ds', basedir='models/dev')
    model.train(all_train_x, all_train_y,
                validation_data=(all_test_x, all_test_y),
                # augmenter=data_augmentation,
                epochs=100)


if __name__ == '__main__':
    main()
