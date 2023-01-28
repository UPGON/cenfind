import sys
from pathlib import Path

import pandas as pd

from cenfind.core.constants import UNITS
from cenfind.core.data import Dataset
from cenfind.core.measure import field_score_frequency


def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "analyse",
        help="Analyse the scoring and compute summary table and visualisation",
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset")
    parser.add_argument(
        "--by", type=str, required=True, help="Grouping (field or well)"
    )

    return parser


def run(args):
    dataset = Dataset(args.dataset)

    if args.by not in UNITS:
        print(f"ERROR: `by` is not in {UNITS} ({{args.by}})")
        sys.exit(2)

    scores = pd.read_csv(dataset.statistics / "scores_df.tsv", sep="\t")

    try:
        binned = field_score_frequency(scores, by=args.by)
    except ValueError as e:
        print(
            "The field value of `fov` does not conform with the syntax `<WELL>_<FOVID>`(%s) (%s)"
            % (scores.loc[0, "fov"], e)
        )
        sys.exit()

    path_stats = dataset.statistics / "statistics.tsv"
    binned.to_csv(path_stats, sep="\t", index=True)
    print(f"Writing statistics to {path_stats}")

    if args.by != "well":
        return 0

    # data = pd.read_csv(path_stats , sep='\t', header=[0, 1, 2])
    # vmin = 'A1'
    # vmax = 'A4'
    # data = prepare_data(data)
    # summed = reduce_data(data)
    # fig = generate_figure(summed, vmin=vmin, vmax=vmax)
    # fig.savefig(dataset.statistics / 'layout_score.png', dpi=300)
