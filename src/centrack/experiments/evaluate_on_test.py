import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from centrack.cli.score import get_model

from centrack.experiments.constants import datasets, PREFIX_REMOTE, pattern_dataset, protein_names, celltype_names
from centrack.data.base import Dataset, Projection, Channel, Field, extract_info
from spotipy.utils import normalize_fast2d, points_matching

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
        h, w = channel.data.shape
        data = channel.data.astype('uint16')

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

            perfs.append({'dataset': ds.path.name,
                          'fov': fov.name,
                          'channel': channel_id,
                          'foci_actual_n': len(annotation),
                          'foci_preds_n': len(points_preds),
                          'cutoff': cutoff,
                          'f1': res.f1,
                          'precision': res.precision,
                          'recall': res.recall}
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
    performances_df = performances_df.set_index('fov')
    return performances_df


def main():
    model_name = '2022-08-17_17:27:44'
    cutoffs = list(range(6))
    model = get_model(f'models/dev/{model_name}')
    path_perfs = f'/home/buergy/projects/centrack/out/performances_{model_name}.csv'
    performances = []

    for dataset_name in datasets:
        path_dataset = PREFIX_REMOTE / dataset_name
        performance = run_evaluation(path_dataset, model, cutoffs)
        performances.append(performance)

    performances_df = perf2df(performances)
    performances_df.to_csv(path_perfs)


if __name__ == '__main__':
    main()
