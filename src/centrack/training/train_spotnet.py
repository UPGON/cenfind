from pathlib import Path
import json
from typing import Tuple
import numpy as np
import tifffile as tif

from spotipy.spotipy.utils import points_to_prob
from spotipy.spotipy.model import SpotNet, Config

from centrack.layout.dataset import DataSet, FieldOfView
from centrack.layout.constants import datasets, PREFIX


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, 'r') as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def generate_mask(path: Path | str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Generate a mask with points as bright spots
    :param path:
    :param shape:
    :return:
    """
    foci = np.loadtxt(path, dtype=int, delimiter=',')  # in format x, y; origin at top left
    mask = points_to_prob(foci, shape=shape, sigma=1)
    return mask


def generate_image(dataset: DataSet, fov: str, channel_id: int) -> np.ndarray:
    """
    Extract the channel from a projection image.
    :param dataset:
    :param fov:
    :param channel_id:
    :return:
    """

    image_path = str(dataset.projections / f"{fov}_max.tif")
    image = tif.imread(image_path)
    channel = image[channel_id, :, :]
    channel = channel.astype('float32') / channel.max()
    return channel


def load_pairs(dataset: DataSet, split='train'):
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    fovs = dataset.split_images_channel(split)
    channel_id = 2
    for fov, _ in fovs:
        channel_id = int(channel_id)

        image = FieldOfView(dataset.projections / f"{fov}_max.tif").data[channel_id]
        image = image.astype('float32') / image.max()
        channels.append(image)

        foci_path = str(dataset.annotations / 'centrioles' / f"{fov}_max_C{channel_id}.txt")
        mask = generate_mask(foci_path, shape=(2048, 2048))
        masks.append(mask)

    return np.stack(channels), np.stack(masks)


def main():
    config = read_config('models/dev/config.json')
    model = SpotNet(config, name='one_ds', basedir='models/dev')
    dataset_path = PREFIX / datasets[0]
    dataset = DataSet(dataset_path)
    train_x, train_y = load_pairs(dataset, split='train')
    test_x, test_y = load_pairs(dataset, split='test')

    model.train(train_x, train_y,
                validation_data=(test_x, test_y))


if __name__ == '__main__':
    main()
