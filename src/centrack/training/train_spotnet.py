from pathlib import Path
import json
import numpy as np
import tifffile as tif

from spotipy.spotipy.utils import points_to_prob
from spotipy.spotipy.model import SpotNet, Config

from centrack.layout.dataset import DataSet
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

    channels = []
    masks = []

    fovs = dataset.split_images_channel(split)

    for fov, channel_id in fovs:
        channel_id = int(channel_id)
        image_path = str(dataset.projections / f"{fov}_max.tif")
        image = tif.imread(image_path)
        channel = image[channel_id, :, :]
        channel = channel.astype('float32') / channel.max()
        channels.append(channel)

        foci_path = str(dataset.annotations / 'centrioles' / f"{fov}_max_C{channel_id}.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')  # in format x, y; origin at top left
        print(f"N foci: {foci.shape}")
        mask = points_to_prob(foci, shape=channel.shape, sigma=1)
        masks.append(mask)
        print(f"max in mask: {mask.max()}")
        print(f"mean intensity image: {channel.mean()}\nmean intensity mask: {mask.mean()}")

    return np.stack(channels), np.stack(masks)


def main():
    config = read_config('models/dev/config.json')
    model = SpotNet(config, name='all_ds', basedir='models/dev')
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

    model.train(all_train_x, all_train_y,
                validation_data=(all_test_x, all_test_y))


if __name__ == '__main__':
    main()
