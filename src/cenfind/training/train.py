import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.random import seed
from spotipy.model import SpotNet, Config
from tqdm import tqdm

from cenfind.training.config import config_multiscale, transforms
from cenfind.core.data import Dataset

seed(1)
tf.random.set_seed(2)


def read_config(path):
    """
    Read config json file.
    :param path:
    :return:
    """
    with open(path, "r") as config:
        config_dict = json.load(config)
    return Config(**config_dict)


def fetch_all_fields(datasets: list[Dataset]):
    all_train_x = []
    all_train_y = []

    all_test_x = []
    all_test_y = []

    for ds in tqdm(datasets):
        train_x, train_y = ds.load_pairs(ds, split="train", transform=transforms)
        test_x, test_y = ds.load_pairs(ds, split="test")
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
