import itertools as it

import numpy as np
import pandas as pd
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


def field_score_frequency(df, by="field"):
    """
    Count the absolute frequency of number of centriole per well or per field
    :param df: Df containing the number of centriole per nuclei
    :param by: the unit to group by, either `well` or `field`
    :return: Df with absolut frequencies.
    """
    cuts = [0, 1, 2, 3, 4, 5, np.inf]
    labels = "0 1 2 3 4 +".split(" ")

    df = df.set_index(["fov", "channel"])
    result = pd.cut(df["score"], cuts, right=False, labels=labels, include_lowest=True)
    result = result.groupby(["fov", "channel"]).value_counts()
    result.name = "freq_abs"
    result = result.sort_index().reset_index()
    result = result.rename({"score": "score_cat"}, axis=1)
    if by == "well":
        result[["well", "field"]] = result["fov"].str.split("_", expand=True)
        print(result.columns)
        result = result.groupby(["well", "channel", "score_cat"])[["freq_abs"]].sum()
        result = result.reset_index()
        result = result.pivot(index=["well", "channel"], columns="score_cat")
        result.reset_index().sort_values(["channel", "well"])
    else:
        result = result.groupby(["fov", "channel", "score_cat"]).sum()
        result = result.reset_index()
        result = result.pivot(index=["fov", "channel"], columns="score_cat")
        result.reset_index().sort_values(["channel", "fov"])

    return result


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
    fig, axes = plt.subplots(nrows=len(channels), ncols=1, figsize=(7 * len(channels), 5))
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
