import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import tifffile as tf

from cenfind.core.data import Dataset, Field
from cenfind.core.loading import load_foci
from cenfind.core.visualisation import visualisation, create_vignette
from cenfind.core.statistics import evaluate

ROOT_DIR = Path("../../../")

logger = logging.getLogger(__name__)

def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "evaluate", help="Evaluate the model on the test split of the dataset"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")
    parser.add_argument("model", type=Path, help="Path to the model")
    parser.add_argument(
        "--channel_nuclei",
        "-n",
        type=int,
        help="Channel index of nuclei",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        nargs="+",
        default=3,
        help="Distance in pixels below which two points are deemed matching",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=.5,
        help="Confidence."
    )
    parser.add_argument(
        "--performances_file",
        type=Path,
        help="Path of the performance file, STDOUT if not specified",
    )

    return parser


def run(args):
    dataset = Dataset(args.dataset)
    dataset.setup()
    if not any((dataset.annotations / "centrioles").iterdir()):
        logger.error(
            f"The dataset {dataset.path.name} has no annotation. You can run `cenfind predict` instead"
        )
        sys.exit(2)

    if type(args.tolerance) == int:
        tolerances = [args.tolerance]
    else:
        tolerances = args.tolerance

    path_visualisation_model = dataset.visualisation / args.model.name
    path_visualisation_model.mkdir(exist_ok=True)

    from cenfind.core.detectors import extract_foci, extract_nuclei

    perfs = []
    with open(dataset.path / "test.txt", "r") as f:
        pairs = [l.strip("\n").split(",") for l in f.readlines()]

    for field_name, channel in pairs:
        channel = int(channel)
        field = Field(dataset.projections / f"{field_name}.tif")
        annotation = load_foci(dataset.annotations / "centrioles" / f"{field.name}_C{channel}.txt")
        predictions = extract_foci(field, channel, args.model, prob_threshold=args.threshold)
        nuclei = extract_nuclei(field, args.channel_nuclei)

        for tol in tolerances:
            logger.info("Processing %s %s %s" % (field, channel, tol))
            perf = evaluate(field, channel, annotation, predictions, tol, threshold=args.threshold)
            background = create_vignette(field, marker_index=channel, nuclei_index=args.channel_nuclei)
            vis = visualisation(background=background, centrioles=predictions, nuclei=nuclei)
            tf.imwrite(path_visualisation_model / f"{field.name}_C{channel}_pred.png", vis)
            perfs.append(perf)

    performance_df = pd.DataFrame(perfs)
    performance_df = performance_df.set_index("field")

    if args.performances_file:
        performance_df.to_csv(args.performances_file)
        print(performance_df)
        logger.info("Performances have been saved under %s" % args.performances_file)
    else:
        print(performance_df)
        logger.warning("Performances are ONLY displayed, not saved")


if __name__ == "__main__":
    args = argparse.Namespace(dataset=ROOT_DIR/ 'data/dataset_test',
                              model=ROOT_DIR / 'models/master',
                              channel_nuclei=0,
                              tolerance=3,
                              threshold=.5,
                              performances_file=ROOT_DIR / "out/perfs.txt"
                              )
    run(args)
