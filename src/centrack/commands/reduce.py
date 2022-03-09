import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(prog="Centriole counter")
    parser.add_argument('source', type=Path, help="Path to the dataset folder")

    return parser.parse_args()


def cli():
    args = parse_args()
    dataset_path = args.source
    results_path = dataset_path / 'results'

    assigned = pd.read_csv(results_path / 'centrioles.csv')
    scores = assigned.groupby(['fov', 'nucleus']).count()['centriole']
    scores.to_csv(dataset_path / 'scores.csv')


if __name__ == '__main__':
    cli()
