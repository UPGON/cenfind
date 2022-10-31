import pandas as pd
from matplotlib import pyplot as plt
from cenfind.experiments.constants import PREFIX_REMOTE

def main():
    data = pd.read_csv('/data1/centrioles/20221019_ZScore_60X_EtOHvsFA_1/statistics/scores_df.tsv', sep='\t')
    # data.columns = ['fov', 'channel', '0', '1', '2', '3', '4', '+']
    data[['well', 'field']] = data['fov'].str.split('_', expand=True)
    summed_wells = data.groupby(['well', 'channel']).sum()
    fig, axes = plt.subplots(12, 8)
    wells = summed_wells.index.get_level_values(0).unique()
    axes_unravelled = axes.ravel()
    for i, well in enumerate(wells):
        sub = summed_wells.loc[well, :]
        axes_unravelled[i].plot(sub)
    print(0)

if __name__ == '__main__':
    main()
