import pandas as pd
from matplotlib import pyplot as plt

from cenfind.experiments.constants import ROOT_DIR
from cenfind.experiments.constants import protein_positions
pd.options.display.max_columns =50

def main():
    path_data = ROOT_DIR / 'out/performances_20221006_130126.csv'
    data = pd.read_csv(path_data)
    data = data.loc[data['n_preds'] > 0]

    datasets_unique = data.dataset.unique()

    fig, axs = plt.subplots(1, len(datasets_unique), figsize=(5*len(datasets_unique), 5))
    for r, dataset in enumerate(datasets_unique):
        sub = data.loc[data['dataset'] == dataset]
        ax = axs[r]
        for channel in sub.channel.unique():
            marker = protein_positions[dataset][channel]

            ssub = sub.loc[sub['channel'] == channel]
            ssub = ssub.sort_values(by='threshold')
            ax.plot(ssub['recall'], ssub['precision'], ls='-', label=marker)

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title(dataset)
        ax.legend()
    fig.savefig(ROOT_DIR / f'out/checks/precision_recall.png')

if __name__ == '__main__':
    main()
