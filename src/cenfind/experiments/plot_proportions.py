import pandas as pd
from cenfind.core.data import Dataset
from cenfind.core.analytics import fraction_zero, plot_layout, fill_plate
import itertools as it

import numpy as np
from matplotlib import pyplot as plt


def main():
    shape = (8, 12)
    dataset = Dataset("/data1/centrioles/20230106/20230106_60X_Grenier_RPE_Plate2_951")

    data = pd.read_csv(
        dataset.statistics / "statistics.tsv", sep="\t", index_col=[0, 1], header=[0, 1]
    )
    data.columns = data.columns.droplevel(0)
    data["frac_zero"] = data.apply(fraction_zero, axis=1)
    data = data.reset_index()
    channels = data["channel"].unique()
    fig, axes = plt.subplots(nrows=len(channels), ncols=1, figsize=(7*len(channels), 5))

    for c, channel in enumerate(channels):
        try:
            ax = axes[c]
        except TypeError:
            ax = axes
        summed_reshaped = fill_plate(data, channel, shape=shape)
        plot_layout(summed_reshaped, channel, ax, vmin=0, vmax=1)

    fig.suptitle("Fraction of centriole-free cells")
    fig.tight_layout()

    fig.savefig(dataset.statistics / "layout_score.png", dpi=300)


if __name__ == "__main__":
    main()
