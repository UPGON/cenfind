import argparse
import os
from pathlib import Path

from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci, extract_nuclei
from cenfind.core.measure import assign, save_foci, save_nuclei_mask, save_assigned, score_nuclei, assign_centrioles, \
    save_scores, save_assigned_centrioles
from cenfind.core.outline import make_visualisation


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
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.dataset)
    dataset.setup()

    pbar = tqdm(dataset.fields)

    for field in pbar:
        pbar.set_description(f"{field.name}")

        nuclei = extract_nuclei(field, args.channel_nuclei)
        save_nuclei_mask(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.png", nuclei,
                         image=field.data[args.channel_nuclei, ...])

        for channel in args.channel_centrioles:
            centrioles = extract_foci(field=field, foci_model_file=args.model, channel=channel)
            save_foci(dataset.centrioles / f"{field.name}_C{channel}.tsv", centrioles, image=field.data[channel, ...])
            assigned = assign(nuclei, centrioles, vicinity=args.vicinity)
            save_assigned(dataset.assignment / f"{field.name}_C{channel}_matrix.tsv", assigned)

            scores = score_nuclei(assigned, nuclei)
            save_scores(dataset.statistics / f"{field.name}_C{channel}_scores.tsv", scores)
            centrioles_nuclei = assign_centrioles(assigned, nuclei, centrioles)
            save_assigned_centrioles(dataset.statistics / f"{field.name}_C{channel}_assigned.tsv", centrioles_nuclei)

            make_visualisation(dataset.visualisation / f"{field.name}_C{channel}.png", field, channel,
                               args.channel_nuclei, centrioles, nuclei, assigned)

            pbar.set_postfix({"field": field.name, "channel": channel,
                              "nuclei": len(nuclei), "foci": len(centrioles)})


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/dataset_test'),
                              model=Path('models/master'),
                              channel_nuclei=0,
                              channel_centrioles=[1, 2],
                              vicinity=50,
                              cpu=False,
                              )
    run(args)
