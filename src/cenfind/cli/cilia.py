import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_cilia, extract_nuclei


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
    pbar = tqdm(dataset.fields)

    proportions = []
    for field in pbar:
        pbar.set_description(f"{field.name}")
        nuclei = extract_nuclei(field, args.channel_nuclei)
        path_cilia = (dataset.visualisation / "cilia")
        path_cilia.mkdir(exist_ok=True)
        cilia = extract_cilia(field, channel=2, dst=path_cilia)
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

    proportions_df = pd.DataFrame(proportions)
    proportions_df.to_csv(dataset.statistics / "proportions_df.tsv", sep="\t", index=False)


if __name__ == "__main__":
    args = argparse.Namespace(dataset=Path('data/cilia'),
                              channel_nuclei=0,
                              channel_cilia=2,
                              cpu=False,
                              )
    run(args)
