import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.outline import create_vignette


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "vignettes",
        help="VIGNETTE: create png version of channel+nuclei for annotation tool",
    )
    parser.add_argument("path", type=Path, help="the path to the dataset")
    parser.add_argument(
        "--channel_nuclei",
        type=int,
        required=True,
        help="the index of the nuclei (often, 0 or 3, first or last)",
    )
    parser.add_argument(
        "--channel_centrioles",
        type=int,
        required=True,
        nargs="+",
        help="the index of the channels",
    )
    parser.add_argument(
        "--projection_suffix",
        type=str,
        default="",
        help="the suffix indicating projection, e.g., `_max` or `_Projected`, empty if not specified",
    )

    return parser


def run(args):
    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)

    pbar = tqdm(dataset.fields)
    for field in pbar:
        for channel_id in args.channel_centrioles:
            pbar.set_description(f"{field.name}: {channel_id}")
            vignette = create_vignette(field, channel_id, args.channel_nuclei)
            dst = (
                dataset.vignettes
                / f"{field.name}{args.projection_suffix}_C{channel_id}.png"
            )
            cv2.imwrite(str(dst), vignette)
