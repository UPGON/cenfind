import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import numpy as np


def fraction_zero(x):
    return x['0'] / x[list('01234+')].sum()


def prepare_data(data):
    data.columns = ['fov', 'channel', '0', '1', '2', '3', '4', '+']
    data[['well', 'field']] = data['fov'].str.split('_', expand=True)
    data = data.set_index(['well', 'field', 'channel'])
    data = data.drop(['fov'], axis=1)
    return data


def reduce_data(data):
    summed = data.groupby(['well', 'channel']).sum()
    summed['frac_zero'] = summed.apply(fraction_zero, axis=1)
    summed = summed.reset_index()
    return summed


def plot_layout(data, channel, ax):
    rows, cols = data.shape
    ax.set_title(f"Channel {channel}")
    ax.imshow(data, cmap='cividis')
    ax.set_xticks(np.arange(cols), labels=[str(i) for i in range(cols)])
    ax.set_yticks(np.arange(rows), labels=list('abcdefgh'))

    for i, j in it.product(range(rows), range(cols)):
        ax.text(j, i, data[i, j].round(2),
                ha='center', va='center', color='w')

    return ax


def reshape_data(data, channel, shape):
    rows, cols = shape
    return (data
            .loc[data['channel'] == channel, 'frac_zero']
            .to_numpy()
            .reshape(rows, cols))

def generate_figure(data):
    shape = (8, 12)
    channels = data['channel'].unique()
    fig, axes = plt.subplots(nrows=len(channels), ncols=1, figsize=(5, 7))
    for c, channel in enumerate(channels):
        ax = axes[c]
        summed_reshaped = reshape_data(data, channel, shape=shape)
        plot_layout(summed_reshaped, channel, ax)

    fig.suptitle('Fraction of centriole-free cells')
    fig.tight_layout()
    return fig

def main():
    path = '/data1/centrioles/20221019_ZScore_60X_EtOHvsFA_1/statistics/statistics.tsv'
    data = pd.read_csv(path, sep='\t', header=[0, 1, 2])

    data = prepare_data(data)
    summed = reduce_data(data)
    fig = generate_figure(summed)
    fig.savefig('out/layout_score.png', dpi=300)


if __name__ == '__main__':
    main()
