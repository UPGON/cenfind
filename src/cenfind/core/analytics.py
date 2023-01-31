import itertools as it

import numpy as np
from matplotlib import pyplot as plt


def str_to_tuple(position):
    row = ord(position[:1].upper()) - 65
    col = int(position[1:])
    return row, col


def fraction_zero(x):
    frac = x["0"] / x[list("01234+")].sum()
    return round(frac, 3)


def plot_layout(data, channel, ax, vmin=0, vmax=1):
    """
    Build heatmap for one channel
    """
    if vmin is None:
        vmin = str_to_tuple(vmin)
    if vmax is None:
        vmax = str_to_tuple(vmax)

    rows, cols = data.shape

    rows_labels = list("abcdefgh".upper())
    cols_labels = [str(i + 1) for i in range(cols)]

    ax.set_title(f"Channel {channel}")
    ax.imshow(data, cmap="cividis", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(cols), labels=cols_labels)
    ax.set_yticks(np.arange(rows), labels=rows_labels)

    for i, j in it.product(range(rows), range(cols)):
        ax.text(j, i, data[i, j].round(2), ha="center", va="center", color="w")

    return ax


def fill_plate(data, channel, shape):
    rows, cols = shape
    for entry in data.iterrows():
        ...
    return (
        data.loc[data["channel"] == channel, "frac_zero"].to_numpy().reshape(rows, cols)
    )


def generate_figure(data, vmin=0, vmax=1):
    shape = (8, 12)
    channels = data["channel"].unique()
    fig, axes = plt.subplots(nrows=len(channels), ncols=1, figsize=(7*len(channels), 5))
    for c, channel in enumerate(channels):
        try:
            ax = axes[c]
        except TypeError:
            ax = axes
        summed_reshaped = fill_plate(data, channel, shape=shape)
        plot_layout(summed_reshaped, channel, ax, vmin=vmin, vmax=vmax)

    fig.suptitle("Fraction of centriole-free cells")
    fig.tight_layout()
    return fig
