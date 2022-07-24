from pathlib import Path
import json
import numpy as np
import tifffile as tif

from spotipy.spotipy.utils import points_to_prob
from spotipy.spotipy.model import SpotNet, Config

from centrack.layout.dataset import DataSet


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

    fovs = dataset.split_images_channel(split)
    images = []
    positions = []
    for fov, channel_id in fovs:
        channel_id = int(channel_id)
        image_path = str(dataset.projections / f"{fov}_max.tif")
        image = tif.imread(image_path)
        image = image[channel_id, :, :]
        image_norm = image.astype('float32') / image.max()
        images.append(image_norm)

        foci_path = str(dataset.annotations / 'centrioles' / f"{fov}_max_C{channel_id}.txt")
        foci = np.loadtxt(foci_path, dtype=int, delimiter=',')  # in format x, y; origin at top left

        foci_mask = points_to_prob(foci, shape=image.shape, sigma=1)
        positions.append(foci_mask)

    return np.stack(images), np.stack(positions)


if __name__ == '__main__':
    dataset = DataSet(Path('/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1'))
    train_x, train_y = load_pairs(dataset, split='train')
    test_x, test_y = load_pairs(dataset, split='test')

    config = read_config('models/dev/config.json')

    model = SpotNet(config, name='model_ds1', basedir='/Users/buergy/Dropbox/epfl/projects/centrack/models/dev')
    model.train(train_x, train_y, validation_data=(test_x, test_y))
