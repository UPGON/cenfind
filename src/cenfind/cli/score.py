import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci, extract_nuclei, extract_cilia
from cenfind.core.measure import Assigner
from cenfind.core.serialise import (
    save_points,
    save_contours,
    save_assigned,
    save_assigned_centrioles,
    save_visualisation
)
from cenfind.core.statistics import proportion_cilia, frequency
from cenfind.core.visualisation import visualisation, create_vignette

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        default=[],
        help="Channel indices to analyse, e.g., 1 2 3",
    )
    parser.add_argument(
        "--channel_cilia",
        "-l",
        type=int,
        help="Channel indices to analyse cilium",
    )
    parser.add_argument(
        "--vicinity",
        type=int,
        default=50,
        help="Distance threshold in pixel (default: 50 px)",
    )

    parser.add_argument("--cpu", action="store_true", help="Only use the cpu")
    parser.add_argument("--verbose", "-v", action="store_true", help="Use logging instead of progress bar")

    return parser


def run(args):
    if (args.channel_centrioles is None) and (args.channel_cilia is None):
        raise ValueError("Please specify at least one channel to evaluate.")

    dataset = Dataset(args.dataset)
    dataset.setup()
    if args.verbose:
        pbar = dataset.fields
    else:
        pbar = tqdm(dataset.fields)
    logger.info("Num GPUs Available: %s" % len(tf.config.list_physical_devices('GPU')))
    if args.cpu:
        logger.info("Using only CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ciliated_container = []
    results = {}

    for field in pbar:
        logger.info("Processing field %s" % field.name)
        if not args.verbose:
            pbar.set_description(f"{field.name}")

        if field.data.ndim != 3:
            logger.error("Image (%s) is not in CXY format (Actual shape: %s)" % (field.name, field.data.shape))
            continue

        channels_actual = set(range(field.data.shape[0]))

        channel_centrioles = set(args.channel_centrioles)
        if not channel_centrioles.issubset(channels_actual):
            logger.warning(
                "Channel %s is beyond the channel span (%s) (Field shape: %s). It has been dismissed" % (
                    channel_centrioles.difference(channels_actual), list(range(field.data.shape[0])), field.data.shape))
            channel_centrioles = list(channel_centrioles.intersection(channels_actual))
        else:
            channel_centrioles = list(channel_centrioles)

        if args.channel_nuclei not in channels_actual:
            logger.warning(
                "channel index (%s) for nuclei not in channel span (%s)" % (args.channel_nuclei, channels_actual))

        nuclei = extract_nuclei(field, args.channel_nuclei)
        if len(nuclei) == 0:
            logger.warning("No nuclei in %s" % field.name)
            continue
        save_contours(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.json", nuclei)

        if not args.verbose:
            pbar_dict = {"nuclei": len(nuclei)}
        if channel_centrioles is not None:
            for channel in channel_centrioles:
                if not args.verbose:
                    pbar_dict["channel"] = channel
                centrioles = extract_foci(field=field, channel=channel, foci_model_file=args.model)
                assignment = Assigner(centrioles, nuclei, vicinity=args.vicinity)
                centrioles_nuclei = assignment.assign_centrioles()
                scores = assignment.score_nuclei(field.name, channel)

                background = create_vignette(field, marker_index=channel, nuclei_index=args.channel_nuclei)
                vis = visualisation(background, centrioles=centrioles, nuclei=nuclei, assigned=assignment.assignment)

                results[(field.name, channel)] = {
                    'scores': scores,
                    'assignment': assignment.assignment,
                    'centrioles_nuclei': centrioles_nuclei,
                    'centrioles': centrioles,
                    'nuclei': nuclei,
                    'visualisation': vis}

                if not args.verbose:
                    pbar_dict[f"centrioles"] = len(centrioles)
                    pbar.set_postfix(pbar_dict)

        if args.channel_cilia is not None:
            channel = args.channel_cilia
            ciliae = extract_cilia(field, channel=channel)
            record = proportion_cilia(field, ciliae, nuclei, channel)
            ciliated_container.append(record)

            save_points(dataset.cilia / f"{field.name}_C{channel}.tsv",
                        ciliae)

            if not args.verbose:
                pbar_dict["cilia"] = len(ciliae)
                pbar.set_postfix(pbar_dict)

    for (field, channel), data in results.items():
        save_points(dataset.centrioles / f"{field}_C{channel}.tsv", data['centrioles'])
        save_assigned(dataset.assignment / f"{field}_C{channel}_matrix.txt", data['assignment'])
        if data['centrioles_nuclei']:
            save_assigned_centrioles(dataset.statistics / f"{field}_C{channel}_assigned.tsv", data['centrioles_nuclei'])
        save_visualisation(dataset.visualisation / f"{field}_C{channel}.png", data['visualisation'])

    if not results:
        raise ValueError("Nothing was detected in the dataset %s." % dataset.path)

    scores_all = pd.concat([v['scores'] for v in results.values()])
    binned = frequency(scores_all)
    binned.to_csv(dataset.statistics / "statistics.tsv", sep="\t", index=True)

    if ciliated_container:
        ciliated_all = pd.concat(ciliated_container)
        ciliated_all.to_csv(dataset.statistics / "ciliated.tsv", sep="\t", index=True)


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              channel_nuclei=0,
                              channel_centrioles=[1, 2, 3],
                              channel_cilia=None,
                              vicinity=50,
                              cpu=True,
                              verbose=True
                              )
    run(args)
