import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import numpy as np

def fraction_zero(x):
    return x['0'] / x[list('01234+')].sum()

def prepare_data(path):
    data = pd.read_csv('', sep='\t', header=[0, 1, 2])
    data.columns = ['fov', 'channel', '0', '1', '2', '3', '4', '+']
    data[['well', 'field']] = data['fov'].str.split('_', expand=True)
    data = data.set_index(['well', 'field', 'channel'])
    data = data.drop(['fov'], axis=1)
    return data

def main():
    path = '/data1/centrioles/20221019_ZScore_60X_EtOHvsFA_1/statistics/statistics.tsv'
    data = prepare_data(path)

    rows, cols = (8, 12)
    summed = data.groupby(['well', 'channel']).sum()
    summed['frac_zero'] = summed.apply(fraction_zero, axis=1)
    summed = summed.reset_index()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
    for c, channel in enumerate(summed['channel'].unique()):
        ax = axes[c]
        ax.set_title(f"Channel {channel}")
        summed_reshaped = (summed
                           .loc[summed['channel'] == channel, 'frac_zero']
                           .to_numpy()
                           .reshape(rows, cols))
        ax.imshow(summed_reshaped, cmap='cividis')
        ax.set_xticks(np.arange(cols), labels=[str(i) for i in range(cols)])
        ax.set_yticks(np.arange(rows), labels=list('abcdefgh'))
        for i, j in it.product(range(rows), range(cols)):
                ax.text(j, i, summed_reshaped[i, j].round(2),
                        ha='center', va='center', color='w')

    fig.suptitle('Fraction of centriole-free cells')
    fig.tight_layout()
    fig.savefig('out/layout_score.png', dpi=300)


if __name__ == '__main__':
    main()
