import argparse
import os
from pathlib import Path

import pandas as pd
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
        "-v",
        type=int,
        default=50,
        help="Distance threshold in pixel (default: 50 px)",
    )

    parser.add_argument("--cpu", action="store_true", help="Only use the cpu")

    return parser


def run(args):
    if (args.channel_centrioles is None) and (args.channel_cilia is None):
        raise ValueError("Please specify at least one channel to evaluate.")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dataset = Dataset(args.dataset)
    dataset.setup()

    pbar = tqdm(dataset.fields)

    ciliated_container = []
    results = {}
    for field in pbar:
        pbar.set_description(f"{field.name}")

        nuclei = extract_nuclei(field, args.channel_nuclei)
        save_contours(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.json", nuclei)

        pbar_dict = {"nuclei": len(nuclei)}
        if args.channel_centrioles is not None:
            for channel in args.channel_centrioles:
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
                    'visualisation': vis}

                pbar_dict[f"centrioles"] = len(centrioles)
                pbar.set_postfix(pbar_dict)

        if args.channel_cilia is not None:
            channel = args.channel_cilia
            ciliae = extract_cilia(field, channel=channel)
            record = proportion_cilia(field, ciliae, nuclei, channel)
            ciliated_container.append(record)

            save_points(dataset.cilia / f"{field.name}_C{channel}.tsv",
                      ciliae)

            pbar_dict["cilia"] = len(ciliae)
            pbar.set_postfix(pbar_dict)

    for (field, channel), data in results.items():
        save_points(dataset.centrioles / f"{field}_C{channel}.tsv", data['centrioles'])
        save_assigned(dataset.assignment / f"{field}_C{channel}_matrix.txt", data['assignment'])
        save_assigned_centrioles(dataset.statistics / f"{field}_C{channel}_assigned.tsv", data['centrioles_nuclei'])
        save_visualisation(dataset.visualisation / f"{field}_C{channel}.png", data['visualisation'])

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
                              channel_centrioles=[1, 2],
                              channel_cilia=3,
                              vicinity=50,
                              cpu=False,
                              )
    run(args)
