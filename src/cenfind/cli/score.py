import argparse
import os
from pathlib import Path

import pandas as pd
import tifffile as tif
from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci, extract_nuclei
from cenfind.core.log import get_logger
from cenfind.core.measure import assign, save_foci, measure_signal_foci
from cenfind.core.outline import visualisation
from tqdm import tqdm


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
        "-v",
        type=int,
        default=50,
        help="Distance threshold in pixel (default: 50 px)",
    )

    parser.add_argument("--cpu", action="store_true", help="Only use the cpu")

    return parser


def run(args):
    if not Path(args.model).exists():
        raise FileNotFoundError(f"{args.model} does not exist")

    if args.channel_nuclei in set(args.channel_centrioles):
        raise ValueError("Nuclei channel cannot be in channels")

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.dataset)

    logger = get_logger(__name__, file=dataset.logs / 'cenfind.log')

    channels = dataset.fields[0].data.shape[0]

    if args.channel_nuclei not in range(channels):
        logger.error("Index for nuclei (%s) out of index range" % args.channel_nuclei, exc_info=True)
        raise ValueError

    if not set(args.channel_centrioles).issubset(set(range(channels))):
        logger.error(
            "Channels (%s) out of channel range %s" % args.channel_centrioles,
            set(range(channels)),
        )
        raise ValueError

    path_visualisation_model = dataset.visualisation / args.model.name
    path_visualisation_model.mkdir(exist_ok=True)

    pbar = tqdm(dataset.fields)
    from cenfind.core.measure import score

    scores = []
    for field in pbar:
        pbar.set_description(f"{field.name}")
        logger.info("Processing %s" % field.name)

        nuclei = extract_nuclei(field, args.channel_nuclei)
        if len(nuclei) == 0:
            logger.warning("No nucleus has been detected in %s" % field.name)
            continue

        for channel in args.channel_centrioles:
            logger.info("Processing %s / %d" % (field.name, channel))
            foci = extract_foci(field=field, foci_model_file=args.model, channel=channel)

            nuclei_scored = assign(nuclei, foci, vicinity=args.vicinity)
            scored = score(field, nuclei_scored, channel)
            scores.append(scored)

            predictions_path = dataset.predictions / "centrioles" / f"{field.name}_C{channel}.txt"
            measurements_path = dataset.statistics / f"{field.name}_C{channel}.txt"
            measure_signal_foci(field, channel, foci, measurements_path)
            save_foci(foci, predictions_path)

            pbar.set_postfix(
                {
                    "field": field.name,
                    "channel": channel,
                    "nuclei": len(nuclei),
                    "foci": len(foci),
                }
            )

            logger.info(
                "(%s), channel %s: nuclei: %s; foci: %s"
                % (field.name, channel, len(nuclei), len(foci))
            )

            vis = visualisation(field, channel_centrioles=channel, channel_nuclei=args.channel_nuclei,
                                nuclei=nuclei_scored)
            tif.imwrite(
                path_visualisation_model / f"{field.name}_C{channel}_pred.png", vis
            )

        logger.info("DONE (%s)" % field.name)

    flattened = [leaf for tree in scores for leaf in tree]
    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / "scores_df.tsv", sep="\t", index=False)
    logger.info("Writing raw scores to %s" % str(dataset.path / "statistics" / "scores_df.tsv"))


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              channel_nuclei=0,
                              channel_centrioles=[1, 2],
                              vicinity=50,
                              cpu=False,
                              )
    run(args)
