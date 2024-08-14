import argparse
import logging

import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

from cenfind.core.data import Dataset
logging.basicConfig(level=logging.ERROR)

def register_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        "analyse",
        help="Analyse predictions",
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset folder")

    return parser

def run(args):

    sns.set_theme()
    dataset = Dataset(args.dataset)
    # Concatenate the predictions and keep identity of field of view
    files = Path(dataset.predictions / "centrioles").glob('*.tsv')
    dfs = list()
    for f in files:
        data = pd.read_csv(f, sep='\t')
        data['file'] = f.stem
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True)

    # Plot the intensities
    g = sns.displot(df, x='intensity', hue='channel')
    g.savefig(dataset.statistics / "vis_intensities.png")

    # Plot the scores
    data = pd.read_csv(dataset.statistics / "statistics.tsv", sep='\t')
    summary = data.groupby('channel').sum(numeric_only=True)
    summary = pd.wide_to_long(summary.reset_index(), stubnames='', i='channel', j='number', suffix='(\d)|\+')
    summary.columns = ['frequency']
    summary = summary.reset_index()
    g = sns.catplot(summary, x='number', y='frequency', hue='channel', kind='bar')
    g.savefig(dataset.visualisation / 'vis_scores.png')

    # Plot SA/I of nuclei
    files = Path(dataset.predictions / 'nuclei').glob('*.json')
    dfs = list()
    for f in files:
        data = pd.read_json(f)
        data = pd.DataFrame(data["nuclei"].values.tolist()).drop("contour", axis=1)
        data['file'] = f.stem
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[df['is_nucleus_full']]

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.scatterplot(data=df, x='surface_area', y='intensity', ax=ax)
    sns.rugplot(data=df, x='surface_area', y='intensity', ax=ax)
    f.savefig(dataset.visualisation / 'vis_surface_area_intensity.png')

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.histplot(data=df, x='intensity', ax=ax, kde=True)
    f.savefig(dataset.visualisation / 'vis_intensity.png')


if __name__ == '__main__':
    args = argparse.Namespace(dataset=Path("../../../data/dataset_test"))
    run(args)
