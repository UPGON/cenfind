import sys
import argparse
from pathlib import Path

import pandas as pd
import tifffile as tf

from cenfind.core.data import Dataset
from cenfind.core.visualisation import visualisation
from cenfind.core.log import get_logger


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
    logger = get_logger(__name__, console=1, file=dataset.logs / "cenfind.log")
    if not any(dataset.path_annotations_centrioles.iterdir()):
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

    from cenfind.core.measure import evaluate
    from cenfind.core.detectors import extract_foci, extract_nuclei

    perfs = []
    pairs = dataset.splits()
    for field, channel in pairs["test"]:
        annotation = field.annotation(channel)
        predictions = extract_foci(field, channel, args.model, prob_threshold=args.threshold)
        nuclei = extract_nuclei(field, args.channel_nuclei)

        for tol in tolerances:
            logger.info("Processing %s %s %s" % (field, channel, tol))
            perf = evaluate(field, channel, annotation, predictions, tol, threshold=args.threshold)
            vis = visualisation(field=field, channel_centrioles=channel, nuclei=nuclei,
                                channel_nuclei=args.channel_nuclei)
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
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              channel_nuclei=0,
                              tolerance=3,
                              threshold=.5,
                              performances_file="out/perfs.txt"
                              )
    run(args)
