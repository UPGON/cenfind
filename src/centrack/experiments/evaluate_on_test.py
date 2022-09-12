from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from spotipy.utils import normalize_fast2d, points_matching

from centrack.cli.score import get_model
from centrack.data.base import Dataset, Projection, Channel, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE

font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}
matplotlib.rc('font', **font)


def run_evaluation(path, model, cutoffs):
    ds = Dataset(path)
    ds.visualisation.mkdir(exist_ok=True)
    test_files = ds.splits_for('test')
    perfs = []

    for fov_name, channel_id in test_files:
        print(fov_name)
        channel_id = int(channel_id)
        fov = Projection(ds, Field(fov_name))
        channel = Channel(fov, channel_id)
        h, w = channel.projection.shape
        data = channel.projection.astype('uint16')

        inp = np.zeros((2048, 2048), dtype='uint16')
        inp[:h, :w] = data
        inp = normalize_fast2d(inp)

        annotation = channel.annotation()
        mask_preds, points_preds = model.predict(inp,
                                                 prob_thresh=.5,
                                                 min_distance=2)

        for cutoff in cutoffs:
            res = points_matching(annotation[:, [1, 0]],
                                  points_preds,
                                  cutoff_distance=cutoff)

            perfs.append({
                'dataset': fov.dataset.path.name,
                'field': fov.field.name,
                'channel': channel_id,
                'foci_actual_n': len(annotation),
                'foci_preds_n': len(points_preds),
                'tolerance': cutoff,
                'precision': res.precision,
                'recall': res.recall,
                'f1': res.f1,
            }
            )
    return perfs


def perf2df(performances) -> pd.DataFrame:
    """
    Convert all performances in a dataframe
    :param performances:
    :return:
    """
    perfs_flat = [s
                  for p in performances
                  for s in p]
    performances_df = pd.DataFrame(perfs_flat).round(3)
    performances_df = performances_df.set_index('field')
    return performances_df


def main():
    model_name = '2022-09-05_09:19:37'
    cutoffs = list(range(6))
    model = get_model(f'models/dev/{model_name}')
    path_out = Path('out')
    path_perfs = path_out / f'performances_{model_name}.csv'
    path_perfs_3px = path_out / f'performances_{model_name}_3px.csv'
    path_summary = path_out / f'performances_{model_name}_3px_summary.csv'
    performances = []

    for dataset_name in datasets:
        path_dataset = PREFIX_REMOTE / dataset_name
        performance = run_evaluation(path_dataset, model, cutoffs)
        performances.append(performance)

    performances_df = perf2df(performances)
    performances_df.to_csv(path_perfs)
    performances_df_3px = performances_df.loc[performances_df["tolerance"] == 3]
    # performances_df_3px.columns = [col.replace('_', ' ').upper() for col in performances_df_3px.columns]
    performances_df_3px.to_csv(path_perfs_3px)
    summary = performances_df_3px.groupby(['dataset', 'channel']).agg('mean', 'std')['f1']
    summary.to_csv(path_summary)


if __name__ == '__main__':
    main()
