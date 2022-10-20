import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from cenfind.experiments.constants import ROOT_DIR


def main():
    path_data = ROOT_DIR / 'out/performances_20221006_130126.csv'
    data = pd.read_csv(path_data)
    sub = data.loc[data['tolerance'] == 5]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(sub['recall'], sub['precision'])
    fig.savefig(ROOT_DIR / 'out/precision_recall_plot.png')
    print(0)


if __name__ == '__main__':
    main()
