import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci, extract_nuclei, extract_cilia
from cenfind.core.measure import (
    assign,
    score_nuclei,
    proportion_cilia,
    assign_centrioles,
    frequency
)
from cenfind.core.serialise import (
    save_foci,
    save_nuclei_mask,
    save_nuclei,
    save_nuclei_contour,
    save_assigned,
    save_assigned_centrioles,
    save_visualisation
)


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

    scores_container = []
    ciliated_container = []
    for field in pbar:
        pbar.set_description(f"{field.name}")

        nuclei = extract_nuclei(field, args.channel_nuclei)
        save_nuclei_mask(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.png", nuclei,
                         image=field.data[args.channel_nuclei, ...])
        save_nuclei(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.txt", nuclei,
                    image=field.data[args.channel_nuclei, ...])
        save_nuclei_contour(dataset.nuclei / f"{field.name}_C{args.channel_nuclei}.json", nuclei)

        dict_pbar = {"nuclei": len(nuclei)}
        if args.channel_centrioles is not None:
            for channel in args.channel_centrioles:
                dict_pbar["channel"] = channel
                centrioles = extract_foci(field=field, channel=channel, foci_model_file=args.model)
                assigned = assign(nuclei, centrioles, vicinity=args.vicinity)

                scores = score_nuclei(assigned, nuclei, field.name, channel)
                scores_container.append(scores)
                centrioles_nuclei = assign_centrioles(assigned, nuclei, centrioles)

                save_foci(dataset.centrioles / f"{field.name}_C{channel}.tsv",
                          centrioles, image=field.data[channel, ...])
                save_assigned(dataset.assignment / f"{field.name}_C{channel}_matrix.txt",
                              assigned)
                save_assigned_centrioles(dataset.statistics / f"{field.name}_C{channel}_assigned.tsv",
                                         centrioles_nuclei)

                save_visualisation(dataset.visualisation / f"{field.name}_C{channel}.png", field, channel,
                                   args.channel_nuclei, centrioles, nuclei, assigned)
                dict_pbar[f"centrioles"] = len(centrioles)
                pbar.set_postfix(dict_pbar)

        if args.channel_cilia is not None:
            channel = args.channel_cilia
            ciliae = extract_cilia(field, channel=channel)
            record = proportion_cilia(field, ciliae, nuclei, channel)
            ciliated_container.append(record)

            save_foci(dataset.cilia / f"{field.name}_C{channel}.tsv",
                      ciliae, image=field.data[channel, ...])
            save_visualisation(dataset.visualisation / f"{field.name}_C{args.channel_cilia}.png", field, channel,
                               args.channel_nuclei, ciliae, nuclei)
            dict_pbar["cilia"] = len(ciliae)
            pbar.set_postfix(dict_pbar)

    ciliated_all = pd.concat(ciliated_container)
    ciliated_all.to_csv(dataset.statistics / "ciliated.tsv", sep="\t", index=True)

    scores_all = pd.concat(scores_container)
    binned = frequency(scores_all)
    binned.to_csv(dataset.statistics / "statistics.tsv", sep="\t", index=True)


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
