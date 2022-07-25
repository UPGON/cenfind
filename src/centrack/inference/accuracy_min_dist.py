import pandas as pd
from matplotlib import pyplot as plt

from centrack.layout.constants import datasets


def main():
    data = pd.read_csv('out/performances_master.csv')
    print(data)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for ds in datasets:
        sub = data.loc[data['ds'] == ds]
        sub_mean = sub.groupby('cutoff').mean().reset_index()
        ax.plot(sub_mean['cutoff'], sub_mean['f1'], color='red')
        ax.plot(sub['cutoff'], sub['f1'], color='red', ls='none', marker='o', alpha=.2, mew=0)
    ax.set_xlabel('Cutoff [pixel]')
    ax.set_ylabel('Accuracy [F-1 score]')
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('out/accuracy_resolution.png')
    return 0


if __name__ == '__main__':
    main()
