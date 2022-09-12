import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from centrack.data.base import extract_info
from centrack.experiments.constants import datasets, pattern_dataset, protein_names, celltype_names

font = {
    'size': 6
}
matplotlib.rc('font', **font)


def _setup_ax(ax):
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)

    return ax


def plot_setup_accuracy(ax, data, metadata, title, accuracy_thresholds=(0, .5, .75, .9, 1.)):
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
    for channel in data.channel.unique():
        channel = int(channel) - 1
        protein_code = metadata["markers"][channel]
        protein_name = protein_names[protein_code]
        marker_name = f"{protein_name}"
        line_type = line_types[channel]
        sub = data.loc[data['channel'] == channel + 1]
        ax.plot(sub['tolerance'], sub['f1'],
                color='#BF3F3F', ls=line_type, lw=1, label=marker_name)
    ax.set_xlim(0, 5)
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
        print(ds)
        cell_type = metadata[ds]['cell_type']
        title = f"{celltype_names[cell_type]} {metadata[ds]['treatment'] or ''}"
        sub = data.loc[data['dataset'] == ds]
        ax = axs[col]
        ax2 = ax.secondary_xaxis("top", functions=(scaler, scaler))
        ax2.set_xlabel('[nm]')
        plot_setup_accuracy(ax, sub, metadata[ds], title, accuracy_thresholds)
    fig.tight_layout()

    return fig


def main():
    model_name = '2022-09-05_09:19:37'
    path_perfs = f'out/performances_{model_name}.csv'
    metadata = {}
    for dataset_name in datasets:
        metadata[dataset_name] = extract_info(pattern_dataset, dataset_name)

    fig = plot_accuracy(path_perfs, metadata)
    fig.savefig(f'out/accuracy_resolution_{model_name}.png', dpi=300)


if __name__ == '__main__':
    main()
