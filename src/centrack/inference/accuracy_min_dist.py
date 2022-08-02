import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from centrack.utils.constants import datasets

font = {'family': 'Helvetica',

        'weight': 'light',
        'size': 6}

matplotlib.rc('font', **font)


def main():
    accuracy_thresholds = [0, .5, .75, .9, 1.]
    data = pd.read_csv('out/performances_master.csv')
    print(data)
    fig, ax = plt.subplots(1, len(datasets), figsize=(12, 2))
    for col, ds in enumerate(datasets):
        a = ax[col]
        sub = data.loc[data['ds'] == ds]
        sub_mean = sub.groupby('cutoff').mean().reset_index()
        a.plot(sub_mean['cutoff'], sub_mean['f1'], color='red')
        a.plot(sub['cutoff'], sub['f1'], color='red',
               ls='none', marker='o', alpha=.2, mew=0)
        a.set_xlabel('Cutoff [pixel]')
        a.set_ylabel('Accuracy [F-1 score]')
        a.set_ylim(0, 1)
        a.set_yticks(accuracy_thresholds)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_title(' '.join(ds.split('_')[:2]))
    fig.tight_layout()
    fig.savefig('out/accuracy_resolution.png')
    return 0


if __name__ == '__main__':
    main()
