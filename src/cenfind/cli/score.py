import argparse
import copy
import os
import sys
from pathlib import Path

import pandas as pd
import tifffile as tif
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.log import get_logger
from cenfind.core.detectors import extract_foci
from cenfind.core.measure import assign, save_foci

logger = get_logger(__name__)


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "score", help="Score the projections given the channels"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument("model", type=Path, help="Absolute path to the model folder")
    parser.add_argument(
        "--channel_nuclei",
        "-n",
        type=int,
        required=True,
        help="Channel index for nuclei segmentation, e.g., 0 or 3",
    )
    parser.add_argument(
        "--channel_centrioles",
        "-c",
        nargs="+",
        type=int,
        required=True,
        help="Channel indices to analyse, e.g., 1 2 3",
    )
    parser.add_argument(
        "--vicinity",
        "-t",
        type=int,
        default=50,
        help="Distance threshold in pixel (default: 50 px)",
    )

    parser.add_argument("--cpu", action="store_true", help="Only use the cpu")

    return parser


def run(args):
    if not args.dataset.exists():
        raise FileNotFoundError(f"{args.dataset} not found")

    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    if args.channel_nuclei in set(args.channel_centrioles):
        raise ValueError("Nuclei channel cannot be in channels")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.dataset)

    dataset.score(channel_nuclei=args.channel_nuclei,
                  channel_centrioles=args.channel_centrioles,
                  method=extract_foci, model=args.model, vicinity=args.vicinity)


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              channel_nuclei=0,
                              channel_centrioles=[1, 2],
                              vicinity=50,
                              cpu=False)
    run(args)
