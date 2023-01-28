import os
import sys
import contextlib
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import tifffile as tif

from cenfind.core.data import Dataset
from cenfind.core.measure import save_foci
from cenfind.core.outline import save_visualisation


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "score", help="Score the projections given the channels"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument("model", type=Path, help="Absolute path to the model folder")
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
    parser.add_argument(
        "--factor",
        type=int,
        default=256,
        help="Factor to use: given a 2048x2048 image, 256 if 63x; 2048 if 20x:",
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

    if not any(dataset.projections.iterdir()):
        print(
            "The projection folder (%s) is empty.\nPlease ensure you have run `squash` or that you have put the projections under projections/"
            % dataset.projections
        )
        sys.exit()

    channels = dataset.fields[0].projection.shape[0]

    if args.channel_nuclei not in range(channels):
        print("Index for nuclei (%s) out of index range" % args.channel_nuclei)
        sys.exit()

    if not set(args.channel_centrioles).issubset(set(range(channels))):
        print(
            "Channels (%s) out of channel range %s" % args.channel_centrioles,
            set(range(channels)),
        )
        sys.exit()

    from stardist.models import StarDist2D

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model_stardist = StarDist2D.from_pretrained("2D_versatile_fluo")

    scores = []

    pbar = tqdm(dataset.fields)
    from cenfind.core.measure import field_score

    for field in pbar:
        print("Processing %s" % field.name)
        pbar.set_description(f"{field.name}")

        for ch in args.channel_centrioles:
            print("Processing %s / %d" % (field.name, ch))
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
                    / f"{field.name}{dataset.projection_suffix}_C{ch}.txt"
                )
                save_foci(foci, predictions_path)
                print(
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

                print("Writing visualisations for (%s), channel %s" % (field.name, ch))
                vis = save_visualisation(
                    field, foci, ch, nuclei, args.channel_nuclei, assigned
                )
                tif.imwrite(dataset.visualisation / f"{field.name}_C{ch}_pred.png", vis)

            except ValueError as e:
                print("%s (%s)" % (e, field.name))
                continue

        print("DONE (%s)" % field.name)

    flattened = [leaf for tree in scores for leaf in tree]
    scores_df = pd.DataFrame(flattened)
    scores_df.to_csv(dataset.statistics / f"scores_df.tsv", sep="\t", index=False)
    print("Writing raw scores to %s" % dataset.statistics / f"scores_df.tsv")
