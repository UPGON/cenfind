import argparse
import json
import uuid

import numpy as np

from spotipy.utils import points_to_prob, normalize_fast2d
from spotipy.model import SpotNet, Config

from centrack.layout.dataset import DataSet, FieldOfView
from centrack.utils.constants import datasets, PREFIX_REMOTE


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, 'r') as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def load_pairs(dataset: DataSet, split: str = 'train', channel_id: int = 2, sigma: float = 1.5):
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    fovs = dataset.split_images_channel(split)
    channel_id = channel_id

    for fov_name, defined_channel_id in fovs:
        fov = FieldOfView(dataset, fov_name)
        if channel_id is None:
            channel_id = defined_channel_id

        image = fov.load_channel(channel_id)
        foci = fov.load_annotation(channel_id)
        image = normalize_fast2d(image)
        mask = points_to_prob(foci, shape=image.shape, sigma=sigma)

        channels.append(image)
        masks.append(mask)

    return np.stack(channels), np.stack(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='The path to the dataset')
    args = parser.parse_args()
    config = Config(n_channel_in=1,
                    backbone='unet',
                    unet_n_depth=3,
                    unet_pool=4,
                    unet_n_filter_base=64,
                    spot_weight=40,
                    multiscale=True,
                    train_learning_rate=3e-4,
                    train_foreground_prob=1,
                    train_batch_norm=False,
                    train_multiscale_loss_decay_exponent=1,
                    train_patch_size=(512, 512),
                    spot_weight_decay=.0,
                    train_batch_size=2)

    model = SpotNet(config, name=str(uuid.uuid4()), basedir='models/dev')
    dataset = DataSet(args.path)
    train_x, train_y = load_pairs(dataset, split='train', channel_id=2)
    test_x, test_y = load_pairs(dataset, split='test', channel_id=2)

    model.train(train_x, train_y, validation_data=(test_x, test_y))


if __name__ == '__main__':
    main()
