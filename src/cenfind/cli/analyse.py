import argparse
from pathlib import Path

import pandas as pd

from cenfind.core.constants import UNITS
from cenfind.core.data import Dataset
from cenfind.core.log import get_logger
from cenfind.core.measure import field_score_frequency

logger = get_logger(__name__)


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
        logger.error("`%s` is not in %s" % (args.by, UNITS), exc_info=True)
        raise

    scores = pd.read_csv(dataset.statistics / "scores_df.tsv", sep="\t")

    try:
        binned = field_score_frequency(scores, by=args.by)
    except ValueError as e:
            logger.error(e, exc_info=True)
            raise

    path_stats = dataset.statistics / "statistics.tsv"
    binned.to_csv(path_stats, sep="\t", index=True)
    logger.info(f"Writing statistics to {path_stats}")

    if args.by != "well":
        return 0


if __name__ == "__main__":
    args = argparse.Namespace(dataset="data/dataset_test", by="field")
    run(args)
