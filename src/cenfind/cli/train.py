import contextlib
import json
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import albumentations as alb

from spotipy.model import SpotNet, Config
from spotipy.utils import points_to_prob, normalize_fast2d

from cenfind.core.data import Dataset

config_unet = Config(
    n_channel_in=1,
    backbone="unet",
    mode="bce",
    unet_n_depth=3,
    unet_pool=4,
    unet_n_filter_base=64,
    spot_weight=40,
    multiscale=False,
    train_learning_rate=3e-4,
    train_foreground_prob=1,
    train_batch_norm=False,
    train_multiscale_loss_decay_exponent=1,
    train_patch_size=(512, 512),
    spot_weight_decay=0.5,
    train_batch_size=2,
)
config_multiscale = Config(
    n_channel_in=1,
    backbone="unet",
    mode="mae",
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
    spot_weight_decay=0.5,
    train_batch_size=2,
)

transforms = alb.Compose(
    [
        alb.ShiftScaleRotate(scale_limit=0.0),
        alb.Flip(),
        alb.RandomBrightnessContrast(always_apply=True),
        alb.RandomGamma(),
    ]
)


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, "r") as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def load_pairs(
    dataset: Dataset, split: str, sigma: float = 1.5, transform: alb.Compose = None
):
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    for field, channel in dataset.pairs(split):
        print(field.name, channel)
        data = field.channel(channel)
        foci = field.annotation(channel)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            image = normalize_fast2d(data)

        if len(foci) == 0:
            mask = np.zeros(image.shape, dtype="uint16")
        else:
            mask = points_to_prob(
                foci[:, [1, 0]], shape=image.shape, sigma=sigma
            )  # because it works with x, y

        if transform is not None:
            transformed = transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        channels.append(image)
        masks.append(mask)

    return np.stack(channels), np.stack(masks)


def fetch_all_fields(datasets: list[Dataset]):
    all_train_x = []
    all_train_y = []

    all_test_x = []
    all_test_y = []

    for ds in tqdm(datasets):
        train_x, train_y = load_pairs(ds, split="train", transform=transforms)
        test_x, test_y = load_pairs(ds, split="test")
        all_train_x.append(train_x)
        all_train_y.append(train_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)

    all_train_x = np.concatenate(all_train_x, axis=0)
    all_train_y = np.concatenate(all_train_y, axis=0)

    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    return all_train_x, all_train_y, all_test_x, all_test_y


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "train", help="Train a Spotnet model using the datasets"
    )
    parser.add_argument("datasets", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model fit"
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")

    return parser


def run(args):
    datasets = [Dataset(path) for path in args.datasets]

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(datasets)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_multiscale = SpotNet(
        config_multiscale, name=time_stamp, basedir=args.model_path
    )

    model_multiscale.train(
        all_train_x,
        all_train_y,
        validation_data=(all_test_x, all_test_y),
        epochs=args.epochs,
    )
