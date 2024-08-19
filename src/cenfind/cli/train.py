import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

from numpy.random import seed
from spotipy.model import SpotNet, Config
from tensorflow.random import set_seed

from cenfind.core.data import Dataset, Field
from cenfind.core.loading import fetch_all_fields
from cenfind.training.config import config_multiscale, transforms

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
    parser.add_argument("channels", type=int, nargs="+", help="the channel numbers to train on, e.g., 1 2 3")
    parser.add_argument(
        "--model_path", type=Path, required=True, default=Path(".").resolve(), help="Path to the model fit"
    )
    parser.add_argument("--epochs", type=int, required=True, default=1, help="Number of epochs")

    return parser


def write_split(split: List[Tuple[Field, int]], dst: Path) -> None:
    with open(dst, "w") as f:
        for field, channel in split:
            line = f"{field.name},{channel}\n"
            f.write(line)


def run(args):
    dataset = Dataset(args.dataset)

    split_train, split_test = dataset.split_pairs(channels=args.channels)
    if not (dataset.path / "train.txt").exists():
        write_split(split_train, dataset.path / "train.txt")
    if not (dataset.path / "test.txt").exists():
        write_split(split_test, dataset.path / "test.txt")

    all_train_x, all_train_y, all_test_x, all_test_y = fetch_all_fields(dataset, transforms=transforms)
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
    args = argparse.Namespace(dataset=Path("../../../data/RPE1wt_CEP63+CETN2+PCNT_1/"),
                              channels=(1,2,3),
                              model_path=Path("../../../models/"),
                              epochs=100)
    run(args)
