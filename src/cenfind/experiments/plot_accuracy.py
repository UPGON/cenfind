import re
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from cenfind.experiments.constants import datasets, pattern_dataset, protein_names, celltype_names

font = {'size': 6}
matplotlib.rc('font', **font)

matplotlib.use("Agg")

def extract_info(pattern: re, dataset_name: str):
    res = re.match(pattern, dataset_name)
    res_dict = res.groupdict()
    markers = res_dict['markers'].split('+')
    res_dict['markers'] = tuple(markers)

    return res_dict

def _setup_ax(ax):
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)

    return ax


def plot_one_dataset(ax, data, metadata, title, accuracy_thresholds=(0, .5, .75, .9, 1.)):
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


def plot_many_datasets(performances: str, metadata: dict, threshold=.5):
    data = pd.read_csv(performances)
    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(2 * len(datasets), 2))
    accuracy_thresholds = (0, .5, .75, .9, 1.)

    def scaler(x):
        return x * 102.5

    for col, ds in enumerate(datasets):
        cell_type = metadata[ds]['cell_type']
        title = f"{celltype_names[cell_type]} {metadata[ds]['treatment'] or ''}"
        sub = data.loc[(data['dataset'] == ds) & (data['threshold'] == threshold)]
        ax = axs[col]
        ax2 = ax.secondary_xaxis("top", functions=(scaler, scaler))
        ax2.set_xlabel('[nm]')
        plot_one_dataset(ax, sub, metadata[ds], title, accuracy_thresholds)
    fig.tight_layout()

    return fig


def main():
    model_name = '20221116_160118'
    path_perfs = f'out/perfs_{model_name}.csv'
    metadata = {}
    for dataset_name in datasets:
        metadata[dataset_name] = extract_info(pattern_dataset, dataset_name)

    fig = plot_many_datasets(path_perfs, metadata)
    fig.savefig(f'out/accuracy_resolution_{model_name}.png', dpi=300)


if __name__ == '__main__':
    main()
