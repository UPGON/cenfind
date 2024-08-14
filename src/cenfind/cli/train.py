import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from numpy.random import seed
from spotipy.model import SpotNet, Config
from tensorflow.random import set_seed

from cenfind.core.data import Dataset
from cenfind.core.loading import fetch_all_fields
from cenfind.training.config import config_multiscale
from cenfind.constants import datasets

seed(1)
set_seed(2)

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


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "train", help="Train a Spotnet model using the datasets"
    )
    parser.add_argument("datasets", type=Path, nargs="+", help="Path to the dataset")
    parser.add_argument(
        "--model_path", type=Path, required=True, default=Path(".").resolve(), help="Path to the model fit"
    )
    parser.add_argument("--epochs", type=int, required=True, default=1, help="Number of epochs")

    return parser


def run(args):
    datasets = [Dataset(path) for path in args.datasets]

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(datasets)
    logger.debug("Loading %s" % (len(all_train_x)))

    if not args.model_path.exists():
        logger.info("Creating model folder (%s)" % (str(args.model_path)))
        args.model_path.mkdir()

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
    datasets = [Path(f"/data1/centrioles/canonical/{ds}") for ds in datasets]
    args = argparse.Namespace(datasets=datasets,
                              model_path=Path("../../../models/"),
                              epochs=100)
    run(args)
