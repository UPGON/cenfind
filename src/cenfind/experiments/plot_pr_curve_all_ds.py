import pandas as pd
from matplotlib import pyplot as plt

from cenfind.experiments.constants import ROOT_DIR

pd.options.display.max_columns = 50


def precision(x):
    return x['tp'] / (x['tp'] + x['fp'])


def recall(x):
    return x['tp'] / (x['tp'] + x['fn'])


def main():
    run = '20221116_160118'
    path_data = ROOT_DIR / f'out/perfs_{run}.csv'
    data = pd.read_csv(path_data)
    data = data.drop(['precision', 'recall', 'f1'], axis=1)
    data_tol_3 = data.loc[data['tolerance'] == 3]
    grouped = data_tol_3.groupby('threshold')[['tp', 'fp', 'fn']].sum()
    grouped['precision'] = grouped.apply(precision, axis=1)
    grouped['recall'] = grouped.apply(recall, axis=1)
    precision0 = pd.DataFrame([[0, 0, 0, 1, 0]],
                              columns=['tp', 'fp', 'fn', 'precision', 'recall'], index=[1])
    grouped = pd.concat([grouped, precision0]).sort_values('recall')
    grouped = grouped.reset_index()
    grouped = grouped.rename({'index': 'threshold'}, axis=1)
    grouped.to_csv(ROOT_DIR / f'out/pr_data_{run}.csv')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(grouped['recall'], grouped['precision'], marker='.', color='blue',
               alpha=grouped['threshold'].to_numpy())
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title("PR Curve\nall DS; any channel")
    fig.savefig(ROOT_DIR / f'out/checks/precision_recall_all_{run}.png', dpi=300)


if __name__ == '__main__':
    main()
