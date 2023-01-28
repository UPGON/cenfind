import sys
from pathlib import Path

import pandas as pd

from cenfind.core.data import Dataset


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "evaluate", help="Evaluate the model on the test split of the dataset"
    )

    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")
    parser.add_argument("model", type=Path, help="Path to the model")
    parser.add_argument(
        "--performances_file",
        type=Path,
        help="Path of the performance file, STDOUT if not specified",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=3,
        help="Distance in pixels below which two points are deemed matching",
    )

    return parser


def run(args):
    dataset = Dataset(args.dataset)
    if not any(dataset.path_annotations_centrioles.iterdir()):
        print(f"ERROR: The dataset {dataset.path.name} has no annotation. You can run `cenfind predict` instead")
        sys.exit(2)

    from cenfind.core.measure import dataset_metrics
    _, performance = dataset_metrics(
        dataset, split="test", model=args.model, tolerance=args.tolerance, threshold=.5
    )

    performance_df = pd.DataFrame(performance)
    performance_df = performance_df.set_index("field")
    if args.performances_file:
        performance_df.to_csv(args.performance_file)
    else:
        print(performance_df)
