import argparse
import contextlib
import os
import sys
from pathlib import Path

import pandas as pd
import tifffile as tif

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci
from cenfind.core.log import get_logger
from cenfind.core.measure import assign

logger = get_logger(__name__)


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "evaluate", help="Evaluate the model on the test split of the dataset"
    )

    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")
    parser.add_argument("model", type=Path, help="Path to the model")
    parser.add_argument(
        "--performances_file",
        type=Path,
        help="Path of the performance file, STDOUT if not specified",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        nargs="+",
        default=3,
        help="Distance in pixels below which two points are deemed matching",
    )
    parser.add_argument(
        "--channel_nuclei",
        type=int,
        required=True,
        help="Channel index for nuclei segmentation, e.g., 0 or 3",
    )
    parser.add_argument(
        "--channel_centrioles",
        nargs="+",
        type=int,
        required=True,
        help="Channel indices to analyse, e.g., 1 2 3",
    )
    parser.add_argument(
        "--vicinity",
        type=int,
        default=-5,
        help="Distance threshold in micrometer (default: -5 um)",
    )

    return parser


def run(args):
    dataset = Dataset(args.dataset)
    if not any(dataset.annotations_centrioles.iterdir()):
        logger.error("The dataset %s has no annotation. You can run `cenfind predict` instead" % dataset.path.name)
        sys.exit(2)

    _, performance = dataset.dataset_metrics(
        split="test", model=args.model, tolerance=args.tolerance, threshold=0.5
    )
    from stardist.models import StarDist2D

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model_stardist = StarDist2D.from_pretrained("2D_versatile_fluo")

    path_visualisation_model = dataset.visualisation / args.model.name
    path_visualisation_model.mkdir(exist_ok=True)

    for field, channel in dataset.pairs("test"):
        foci = field.extract_centrioles(method=extract_foci, model_path=args.model, channel=channel)
        nuclei = field.extract_nuclei(args.channel_nuclei, 256, model_stardist)
        nuclei_scored = assign(nuclei, foci, vicinity=args.vicinity)
        foci, nuclei, assigned, score = field.score(
            nuclei_scored=nuclei_scored,
            channel=channel,
        )
        vis = field.visualisation(
            nuclei=nuclei,
            channel_nuclei=args.channel_nuclei,
            centrioles=foci,
            channel_centrioles=channel,
        )
        tif.imwrite(path_visualisation_model / f"{field.name}_C{channel}_pred.png", vis)

    performance_df = pd.DataFrame(performance)
    performance_df = performance_df.set_index("field")
    if args.performances_file:
        performance_df.to_csv(args.performances_file)
        print(performance_df)
        print("Performances have been saved under %s" % args.performances_file)
    else:
        print(performance_df)
        print("Performances are ONLY displayed, not saved")


if __name__ == '__main__':
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              performances_file=Path('out/performances.tsv'),
                              channel_nuclei=0,
                              channel_centrioles=[1, 2],
                              tolerance=3,
                              vicinity=50
                              )
    run(args)
