import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from centrack.cli.score import get_model

from centrack.experiments.constants import datasets, PREFIX_REMOTE, pattern_dataset
from centrack.data.base import Dataset, Projection, Channel, Field, extract_info
from spotipy.utils import normalize_fast2d, points_matching

font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}
matplotlib.rc('font', **font)


def run_evaluation(path, model, cutoffs):
    ds = Dataset(path)
    ds.visualisation.mkdir(exist_ok=True)
    test_files = ds.splits_for('test')
    perfs = []

    for fov_name, channel_id in test_files:
        print(fov_name)
        channel_id = int(channel_id)
        fov = Projection(ds, Field(fov_name))
        channel = Channel(fov, channel_id)
        h, w = channel.data.shape
        data = channel.data.astype('uint16')

        inp = np.zeros((2048, 2048), dtype='uint16')
        inp[:h, :w] = data
        inp = normalize_fast2d(inp)

        annotation = channel.annotation()
        mask_preds, points_preds = model.predict(inp,
                                                 prob_thresh=.5,
                                                 min_distance=2)
        for cutoff in cutoffs:
            res = points_matching(annotation[:, [1, 0]],
                                  points_preds,
                                  cutoff_distance=cutoff)

            perfs.append({'ds': ds.path.name,
                          'fov': fov.name,
                          'channel': channel_id,
                          'foci_actual_n': len(annotation),
                          'foci_preds_n': len(points_preds),
                          'cutoff': cutoff,
                          'f1': res.f1,
                          'precision': res.precision,
                          'recall': res.recall}
                         )
    return perfs


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
    ax = _setup_ax(ax)
    for c in data.channel.unique():
        marker_index = int(c) - 1
        marker_name = f"{metadata['markers'][marker_index]}"
        sub = data.loc[data['channel'] == c]
        ax.plot(sub['cutoff'], sub['f1'],
                color='black', ls='-', lw=.5, marker='.', alpha=1, mew=0, label=marker_name)
    ax.set_xlabel('Tolerance [pixel]')
    ax.set_ylabel('Accuracy [F-1 score]')
    ax.set_yticks(accuracy_thresholds)
    ax.set_title(title)
    ax.legend(loc='lower right')
    return ax


def plot_accuracy(performances: str, metadata: dict):
    data = pd.read_csv(performances)
    fig, axs = plt.subplots(1, len(datasets), figsize=(2 * len(datasets), 2))
    accuracy_thresholds = (0, .5, .75, .9, 1.)

    def scaler(x):
        return x * 102.5

    for col, ds in enumerate(datasets):
        title = metadata[ds]['cell_type']
        sub = data.loc[data['ds'] == ds]
        ax = axs[col]
        ax2 = ax.secondary_xaxis("top", functions=(scaler, scaler))
        ax2.set_xlabel('Tolerance [nm]')
        plot_setup_accuracy(ax, sub, title, metadata[ds], accuracy_thresholds)
    fig.tight_layout()

    return fig


def perf2df(performances) -> pd.DataFrame:
    """
    Convert all performances in a dataframe
    :param performances:
    :return:
    """
    perfs_flat = [s
                  for p in performances
                  for s in p]
    performances_df = pd.DataFrame(perfs_flat).round(3)
    performances_df = performances_df.set_index('fov')
    return performances_df


def main():
    model_name = '2022-08-15_13:45:46'
    cutoffs = list(range(6))
    model = get_model(f'models/dev/{model_name}')
    path_perfs = f'/home/buergy/projects/centrack/out/performances_{model_name}.csv'
    performances = []
    metadata = {}

    for dataset_name in datasets:
        metadata[dataset_name] = extract_info(pattern_dataset, dataset_name)
        path_dataset = PREFIX_REMOTE / dataset_name
        performance = run_evaluation(path_dataset, model, cutoffs)
        performances.append(performance)

    performances_df = perf2df(performances)
    performances_df.to_csv(path_perfs)

    fig = plot_accuracy(path_perfs, metadata)
    fig.savefig('out/accuracy_resolution.png', dpi=300)


if __name__ == '__main__':
    main()
