import argparse
from pathlib import Path

import tifffile as tf

from cenfind.core.data import Dataset
from cenfind.core.detectors import extract_foci


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "predict",
        help="Predict centrioles on new dataset and save under visualisations/runs/<model_name>",
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")
    parser.add_argument("model", type=Path, help="Path to the model")
    parser.add_argument(
        "--channel_nuclei", type=int, required=True, help="Index of nuclei channel"
    )

    return parser


def run(args):
    dataset = Dataset(args.dataset)

    model_predictions = dataset.visualisation / "runs"
    model_predictions.mkdir(exist_ok=True)

    model_run = model_predictions / args.model.name
    model_run.mkdir(exist_ok=True)

    for field, channel in dataset.pairs("test"):
        foci = field.extract_centrioles(method=extract_foci, model_path=args.model, channel=channel)
        nuclei = field.extract_nuclei(args.channel_nuclei, 256)
        print(
            "Writing visualisations for field: %s, channel: %s, %s foci detected"
            % (field.name, channel, len(foci))
        )
        vis = field.visualisation(
            nuclei=nuclei, centrioles=foci, channel_centrioles=channel, channel_nuclei=args.channel_nuclei
        )
        tf.imwrite(model_run / f"{field.name}_C{channel}_pred.png", vis)
