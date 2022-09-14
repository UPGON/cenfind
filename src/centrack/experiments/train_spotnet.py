import argparse
import json
import uuid
import albumentations as alb
import numpy as np

from spotipy.utils import points_to_prob, normalize_fast2d
from spotipy.model import SpotNet, Config

from centrack.core.data import Dataset, Field

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
                spot_weight_decay=.5,
                train_batch_size=2)

transforms = alb.Compose([
    alb.ShiftScaleRotate(scale_limit=0.),
    alb.Flip(),
])


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, 'r') as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def load_pairs(dataset: Dataset, split: str, sigma: float = 1.5, transform: alb.Compose = None):
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    fovs = dataset.fields(split)

    for field, channel in fovs:
        field = Field(field, dataset)
        channel = int(channel)
        data = field.channel(channel)
        foci = field.annotation(channel)
        image = normalize_fast2d(data)
        mask = points_to_prob(foci, shape=image.shape, sigma=sigma)

        if transform is not None:
            transformed = transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        channels.append(image)
        masks.append(mask)

    return np.stack(channels), np.stack(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='The path to the dataset')
    args = parser.parse_args()

    model = SpotNet(config, name=str(uuid.uuid4()), basedir='models/dev')
    dataset = Dataset(args.path)
    train_x, train_y = load_pairs(dataset, split='train', transform=transforms)
    test_x, test_y = load_pairs(dataset, split='test')

    model.train(train_x, train_y, validation_data=(test_x, test_y), epochs=100)


if __name__ == '__main__':
    main()
