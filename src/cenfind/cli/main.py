from numpy.random import seed

seed(1)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

tf.random.set_seed(2)

import argparse
import logging
import sys
from datetime import datetime
import contextlib
import os
from pathlib import Path

import cv2
import pandas as pd
import tifffile as tf
from tqdm import tqdm

from cenfind.core.measure import field_score_frequency, save_foci
from cenfind.core.outline import Centre, create_vignette
from cenfind.core.data import Dataset


def cli_prepare(args, logger):
    if not args.path.exists():
        print(f"The path `{args.path}` does not exist.")
        sys.exit()

    path_dataset = args.path
    dataset = Dataset(path_dataset, projection_suffix=args.projection_suffix)

    dataset.setup()
    dataset.write_fields()
    if args.splits:
        dataset.write_train_test(args.splits)


def cli_squash(args, logger):
    dataset = Dataset(args.path)

    if dataset.has_projections:
        print(f"Projections already exist, squashing skipped.")
        sys.exit()

    if not dataset.raw.exists():
        print(f"Folder raw/ not found.")
        sys.exit()

    dataset.write_projections()


def cli_score(args, logger):
    if args.channel_nuclei in set(args.channels):
        raise ValueError("Nuclei channel cannot be in channels")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"{args.model} does not exist")
    visualisation = True

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.path, projection_suffix=args.projection_suffix)
    if not any(dataset.projections.iterdir()):
        logger.error(
            "The projection folder (%s) is empty.\nPlease ensure you have run `squash` or that you have put the projections under projections/"
            % dataset.projections
        )
        sys.exit()

    channels, width, height = dataset.fields[0].projection.shape

    if args.channel_nuclei not in range(channels):
        logger.error("Index for nuclei (%s) out of index range" % args.channel_nuclei)
        sys.exit()

    if not set(args.channels).issubset(set(range(channels))):
        logger.error(
            "Channels (%s) out of channel range %s" % args.channels,
            set(range(channels)),
        )
        sys.exit()

    from stardist.models import StarDist2D

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model_stardist = StarDist2D.from_pretrained("2D_versatile_fluo")

    scores = []
    pbar = tqdm(dataset.fields)
    from cenfind.core.measure import field_score, field_score_frequency

    for field in pbar:
        logger.info("Processing %s" % field.name)
        pbar.set_description(f"{field.name}")
        for ch in args.channels:
            logger.info("Processing %s / %d" % (field.name, ch))
            try:
                foci, nuclei, assigned, score = field_score(
                    field=field,
                    model_nuclei=model_stardist,
                    model_foci=args.model,
                    nuclei_channel=args.channel_nuclei,
                    factor=args.factor,
                    vicinity=args.vicinity,
                    channel=ch,
                )
                predictions_path = (
                    dataset.predictions
                    / "centrioles"
                    / f"{field.name}{args.projection_suffix}_C{ch}.txt"
                )
                save_foci(foci, predictions_path)
                logger.info(
                    "(%s), channel %s: nuclei: %s; foci: %s"
                    % (field.name, ch, len(nuclei), len(foci))
                )
                pbar.set_postfix(
                    {
                        "field": field.name,
                        "channel": ch,
                        "nuclei": len(nuclei),
                        "foci": len(foci),
                    }
                )
                scores.append(score)
                if visualisation:
                    logger.info(
                        "Writing visualisations for (%s), channel %s" % (field.name, ch)
                    )
                    background = create_vignette(
                        field, marker_index=ch, nuclei_index=args.channel_nuclei
                    )
                    for focus in foci:
                        background = focus.draw(background, annotation=False)
                    for nucleus in nuclei:
                        background = nucleus.draw(background, annotation=False)
                    for n_pos, c_pos in assigned:
                        nuc = Centre(n_pos, label="Nucleus")
                        for sub_c in c_pos:
                            if sub_c:
                                cv2.arrowedLine(
                                    background,
                                    sub_c.to_cv2(),
                                    nuc.to_cv2(),
                                    color=(0, 255, 0),
                                    thickness=1,
                                )
                    tf.imwrite(
                        dataset.visualisation / f"{field.name}_C{ch}_pred.png",
                        background,
                    )
            except ValueError as e:
                logger.warning("%s (%s)" % (e, field.name))
                continue

        logger.info("DONE (%s)" % field.name)

    flattened = [leaf for tree in scores for leaf in tree]
    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / f"scores_df.tsv", sep="\t", index=False)
    logger.info("Writing raw scores to scores_df.tsv")
    logger.info("All fields in (%s) have been processed" % dataset.path.name)


def cli_analyse(args, logger):
    dataset = Dataset(args.path)
    scores = pd.read_csv(dataset.statistics / "scores_df.tsv", sep="\t")

    try:
        binned = field_score_frequency(scores, by=args.by)
    except ValueError as e:
        logger.error(
            "The field value of `fov` does not conform with the syntax `<WELL>_<FOVID>`(%s) (%s)"
            % (scores.loc[0, "fov"], e)
        )
        sys.exit()

    binned.to_csv(dataset.statistics / f"statistics.tsv", sep="\t", index=True)
    logger.info("Writing statistics to statistics.tsv")


def main():
    parser = argparse.ArgumentParser(prog="CENFIND")
    subparsers = parser.add_subparsers(title="subprograms")

    # Prepare
    parser_prepare = subparsers.add_parser(
        "prepare", help="Prepare the dataset structure"
    )
    parser_prepare.add_argument("path", type=Path, help="Path to the dataset")
    parser_prepare.add_argument(
        "--projection_suffix",
        type=str,
        default="_max",
        help="Suffix indicating projection, e.g., `_max` (default) or `Projected`",
    )
    parser_prepare.add_argument(
        "--splits",
        type=int,
        nargs="+",
        help="Write the train and test splits for continuous learning using the channels specified",
    )
    parser_prepare.set_defaults(func=cli_prepare)

    # Squash
    parser_squash = subparsers.add_parser("squash", help="Write z-projections.")
    parser_squash.add_argument("path", type=Path, help="Path to the dataset folder")
    parser_squash.set_defaults(func=cli_squash)

    # Score
    parser_score = subparsers.add_parser(
        "score", help="Score the projections given the channels"
    )
    parser_score.add_argument("path", type=Path, help="Path to the dataset")
    parser_score.add_argument(
        "model", type=Path, help="Absolute path to the model folder"
    )
    parser_score.add_argument(
        "channel_nuclei",
        type=int,
        help="Channel index for nuclei segmentation, e.g., 0 or 3",
    )
    parser_score.add_argument(
        "channels", nargs="+", type=int, help="Channel indices to analyse, e.g., 1 2 3"
    )
    parser_score.add_argument(
        "--vicinity",
        type=int,
        default=-5,
        help="Distance threshold in micrometer (default: -5 um)",
    )
    parser_score.add_argument(
        "--factor",
        type=int,
        default=256,
        help="Factor to use: given a 2048x2048 image, 256 if 63x; 2048 if 20x:",
    )
    parser_score.add_argument(
        "--projection_suffix",
        type=str,
        default="",
        help="Projection suffix (`_max` or `_Projected`) if not specified, empty",
    )
    parser_score.add_argument("--cpu", action="store_true", help="Only use the cpu")
    parser_score.set_defaults(func=cli_score)

    # Analyse
    parser_analyse = subparsers.add_parser(
        "analyse",
        help="Analyse the scoring and compute summary table and visualisation",
    )
    parser_analyse.add_argument("path", type=Path, help="Path to the dataset")
    parser_analyse.add_argument("--by", type=str, help="Grouping (field or well)")
    parser_analyse.set_defaults(func=cli_analyse)

    args = parser.parse_args()

    path_logs = args.path / "logs"
    path_logs.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    start_stamp = datetime.now()
    log_file = f'{start_stamp.strftime("%Y%m%d_%H:%M:%S")}_score.log'

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(filename=path_logs / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except PermissionError:
        logger.warning(
            "Could not create %s because of permission; will only log to STDOUT" % log_file
        )

    args.func(args, logger)


if __name__ == "__main__":
    main()
