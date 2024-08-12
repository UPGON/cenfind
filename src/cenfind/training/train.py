import argparse
import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import albumentations as alb
import numpy as np
import tifffile as tf
from numpy.random import seed
from spotipy.model import SpotNet, Config
from spotipy.utils import points_to_prob, normalize_fast2d
from tensorflow.random import set_seed
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.training.config import config_multiscale, transforms

seed(1)
set_seed(2)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        ds: Dataset, split: str, sigma: float = 1.5, transform: alb.Compose = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two arrays, the images and the foci masks
    path: the path to the ds
    split: either train or test
    """

    channels = []
    masks = []

    with open(ds.path / f"{split}.txt", "r") as f:
        pairs = [l.strip("\n").split(",") for l in f.readlines()]

    for field, channel in pairs:
        data = tf.imread(ds.projections / f"{field}_max.tif")[int(channel), :, :]
        foci = np.loadtxt(ds.annotations / "centrioles" / f"{field}_max_C{channel}.txt", dtype=int, delimiter=",")

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            image = normalize_fast2d(data)

        if len(foci) == 0:
            mask = np.zeros(image.shape, dtype="uint16")
        else:
            mask = points_to_prob(
                foci[:, [0, 1]], shape=image.shape, sigma=sigma
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
    parser.add_argument("datasets", type=Path, nargs="+", help="Path to the dataset")
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model fit"
    )
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")

    return parser


def run(args):
    datasets = [Dataset(path) for path in args.datasets]

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(datasets)
    logger.debug("Loading %s" % (len(all_train_x)))

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


if __name__ == "__main__":
    args = argparse.Namespace(datasets=[
        Path("/Users/buergy/Dropbox/epfl/projects/cenfind/data/cenfind_datasets/RPE1p53+Cnone_CEP63+CETN2+PCNT_1")],
                              model_path=Path("/Users/buergy/Dropbox/epfl/projects/cenfind/models/"),
                              epochs=100)
    run(args)
