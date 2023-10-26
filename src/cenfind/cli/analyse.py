import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


def main():
    sns.set_theme()
    working_dir = Path('/Users/buergy/Downloads/IF_2022_09_07_U2OS_WTasync')
    # Concatenate the predictions and keep identity of field of view
    files = Path(working_dir / 'predictions/centrioles').glob('*.tsv')
    dfs = list()
    for f in files:
        data = pd.read_csv(f, sep='\t')
        data['file'] = f.stem
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True)

    # Plot the intensities
    g = sns.displot(df, x='intensity', hue='channel')
    g.savefig(working_dir / 'visualisation/test.png')

    # Plot the scores
    data = pd.read_csv(working_dir / 'statistics/statistics.tsv', sep='\t')
    summary = data.groupby('channel').sum(numeric_only=True)
    summary = pd.wide_to_long(summary.reset_index(), stubnames='', i='channel', j='number', suffix='(\d)|\+')
    summary.columns = ['frequency']
    summary = summary.reset_index()
    g = sns.catplot(summary, x='number', y='frequency', hue='channel', kind='bar')
    g.savefig(working_dir / 'visualisation/test2.png')

    # Plot SA/I of nuclei
    files = Path(working_dir / 'predictions/nuclei').glob('*.json')
    dfs = list()
    for f in files:
        data = pd.read_json(f, orient='index')
        data = data.drop('contour', axis=1)
        data['file'] = f.stem
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    df = df.loc[df['is_nucleus_full']]

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.scatterplot(data=df, x='surface_area', y='intensity', ax=ax)
    sns.rugplot(data=df, x='surface_area', y='intensity', ax=ax)
    f.savefig(working_dir / 'visualisation/test3.png')

    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.histplot(data=df, x='intensity', ax=ax, kde=True)
    f.savefig(working_dir / 'visualisation/test4.png')

    print(df)


if __name__ == '__main__':
    main()
