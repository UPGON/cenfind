import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_cilia, extract_nuclei
from cenfind.core.log import get_logger


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "cilia", help="Detect cilia with the Hessian matrix and return the proportion of ciliated cells"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--channel_nuclei",
        "-n",
        type=int,
        required=True,
        help="Channel index for nuclei segmentation, e.g., 0 or 3",
    )
    parser.add_argument(
        "--channel_cilia",
        "-c",
        type=int,
        required=True,
        help="Channel index to analyse, e.g., 2",
    )
    parser.add_argument("--cpu", action="store_true", help="Only use the cpu")

    return parser


def run(args):
    if args.channel_nuclei == args.channel_cilia:
        raise ValueError("Nuclei channel cannot be in channels")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.dataset)
    logger = get_logger(__name__, file=dataset.logs / 'cenfind.log')

    if not any(dataset.projections.iterdir()):
        logger.error(
            "The projection folder (%s) is empty.\nPlease ensure you have run `squash` or that you have put the projections under projections/"
            % dataset.projections, exc_info=True
        )
        raise FileNotFoundError

    channels = dataset.fields[0].projection.shape[0]

    if args.channel_nuclei not in range(channels):
        logger.error("Index for nuclei (%s) out of index range" % args.channel_nuclei, exc_info=True)
        raise ValueError

    if args.channel_cilia not in set(range(channels)):
        logger.error(
            "Channels (%s) out of channel range %s" % args.channel_cilia,
            set(range(channels)),
        )
        raise ValueError

    pbar = tqdm(dataset.fields)

    proportions = []
    for field in pbar:
        pbar.set_description(f"{field.name}")
        logger.info("Processing %s" % field.name)

        nuclei = extract_nuclei(field, args.channel_nuclei, 256)
        if len(nuclei) == 0:
            logger.warning("No nucleus has been detected in %s" % field.name)
            continue
        logger.info("Processing %s / %d" % (field.name, args.channel_cilia))
        path_cilia = (dataset.visualisation / "cilia")
        path_cilia.mkdir(exist_ok=True)
        cilia = extract_cilia(field, channel=2, dst=path_cilia)
        if len(cilia) == 0:
            logger.warning("No centrioles (channel: %s) has been detected in %s" % (args.channel_cilia, field.name))

        ciliated = len(cilia) / len(nuclei)
        proportions.append({'field': field.name,
                            "channel_cilia": args.channel_cilia,
                            "n_nuclei": len(nuclei),
                            "n_cilia": len(cilia),
                            "p_ciliated": round(ciliated, 2)})

        pbar.set_postfix(
            {
                "channel": args.channel_cilia,
                "nuclei": len(nuclei),
                "cilia": len(cilia),
            }
        )

        logger.info(
            "(%s), channel %s: nuclei: %s; foci: %s"
            % (field.name, args.channel_cilia, len(nuclei), len(cilia))
        )

        logger.info(
            "Writing visualisations for (%s), channel %s" % (field.name, args.channel_cilia)
        )

        logger.info("DONE (%s)" % field.name)

    proportions_df = pd.DataFrame(proportions)
    proportions_df.to_csv(dataset.statistics / "proportions_df.tsv", sep="\t", index=False)
    logger.info("Writing raw proportions to %s" % str(dataset.statistics / "proportions_df.tsv"))


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/cilia'),
                              channel_nuclei=0,
                              channel_cilia=2,
                              cpu=False,
                              )
    run(args)
