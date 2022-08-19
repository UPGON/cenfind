import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from centrack.cli.score import get_model

from centrack.experiments.constants import datasets, PREFIX_REMOTE, pattern_dataset, protein_names, celltype_names
from centrack.data.base import Dataset, Projection, Channel, Field, extract_info
from spotipy.utils import normalize_fast2d, points_matching

font = {
    # 'family': 'Helvetica',
    # 'weight': 'light',
    'size': 6
}
matplotlib.rc('font', **font)


def _setup_ax(ax):
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)

    return ax


def plot_setup_accuracy(ax, data, title, metadata, accuracy_thresholds=(0, .5, .75, .9, 1.)):
    """
    Plot the accuracy on the specified ax
    :param metadata:
    :param ax:
    :param data:
    :param title:
    :param accuracy_thresholds:
    :return:
    """
    line_types = ['-', '--', ':']
    ax = _setup_ax(ax)
    for c in data.channel.unique():
        marker_index = int(c) - 1
        protein = metadata['markers'][marker_index]
        marker_name = f"{protein_names[protein]}"
        line_type = line_types[marker_index]
        sub = data.loc[data['channel'] == c]
        ax.plot(sub['cutoff'], sub['f1'],
                color='blue', ls=line_type, lw=1, label=marker_name)
    ax.set_xlim(2, 5)
    ax.set_xlabel('[pixel]\nTolerance')
    ax.set_ylabel('Accuracy [F-1 score]')

    ax.set_yticks(accuracy_thresholds)
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=False)
    return ax


def plot_accuracy(performances: str, metadata: dict):
    data = pd.read_csv(performances)
    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(2 * len(datasets), 2), sharey=True)
    accuracy_thresholds = (0, .5, .75, .9, 1.)

    def scaler(x):
        return x * 102.5

    for col, ds in enumerate(datasets):
        cell_type = metadata[ds]['cell_type']
        title = f"{celltype_names[cell_type]} {metadata[ds]['treatment'] or ''}"
        sub = data.loc[data['dataset'] == ds]
        ax = axs[col]
        ax2 = ax.secondary_xaxis("top", functions=(scaler, scaler))
        ax2.set_xlabel('[nm]')
        plot_setup_accuracy(ax, sub, title, metadata[ds], accuracy_thresholds)
    fig.tight_layout()

    return fig


def main():
    model_name = '2022-08-17_17:27:44'
    path_perfs = f'/home/buergy/projects/centrack/out/performances_{model_name}.csv'
    metadata = {}
    for dataset_name in datasets:
        metadata[dataset_name] = extract_info(pattern_dataset, dataset_name)

    fig = plot_accuracy(path_perfs, metadata)
    fig.savefig(f'out/accuracy_resolution_{model_name}.png', dpi=300)


if __name__ == '__main__':
    main()
