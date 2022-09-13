from pathlib import Path
from typing import List, Dict

import matplotlib
import numpy as np
import pandas as pd
from spotipy.utils import normalize_fast2d, points_matching

from centrack.scoring.detectors import get_model
from centrack.data.base import Dataset, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE

font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}
matplotlib.rc('font', **font)


def run_evaluation(dataset: Dataset, model, tolerance: list[int]):
    test_files = dataset.splits_for('test')

    perfs = []
    for field_name, channel in test_files:
        field = Field(field_name, dataset)
        channel = int(channel)
        inp = normalize_fast2d(field.channel(channel))

        annotation = field.annotation(channel)
        mask_preds, points_preds = model.predict(inp,
                                                 prob_thresh=.5,
                                                 min_distance=2)

        for cutoff in tolerance:
            res = points_matching(annotation[:, [1, 0]],
                                  points_preds,
                                  cutoff_distance=cutoff)

            perfs.append({
                'dataset': dataset.path.name,
                'field': field.name,
                'channel': channel,
                'foci_actual_n': len(annotation),
                'foci_preds_n': len(points_preds),
                'tolerance': cutoff,
                'precision': res.precision.round(3),
                'recall': res.recall.round(3),
                'f1': res.f1.round(3),
            }
            )
    return perfs


def perf2df(performances: List) -> pd.DataFrame:
    """
    Convert all performances in a dataframe
    :param performances:
    :return:
    """
    performances_df = pd.DataFrame([s
                                    for p in performances
                                    for s in p])
    performances_df = performances_df.set_index('field')
    return performances_df


def main():
    model_name = '2022-09-05_09:19:37'
    tolerance = list(range(6))
    model = get_model(f'models/dev/{model_name}')
    path_out = Path('out')
    path_perfs = path_out / f'performances_{model_name}.csv'
    path_perfs_3px = path_out / f'performances_{model_name}_3px.csv'
    path_summary = path_out / f'performances_{model_name}_3px_summary.csv'
    performances = []

    for dataset_name in datasets:
        path_dataset = PREFIX_REMOTE / dataset_name
        dataset = Dataset(path_dataset)
        performance = run_evaluation(dataset, model, tolerance)
        performances.append(performance)

    performances_df = perf2df(performances)
    performances_df.to_csv(path_perfs)

    performances_df_3px = performances_df.loc[performances_df["tolerance"] == 3]
    performances_df_3px.to_csv(path_perfs_3px)

    summary = performances_df_3px.groupby(['dataset', 'channel']).agg('mean', 'std')['f1']
    summary.to_csv(path_summary)


if __name__ == '__main__':
    main()
