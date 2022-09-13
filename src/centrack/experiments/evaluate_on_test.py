from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from spotipy.utils import points_matching

from centrack.data.base import Dataset, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from centrack.scoring.detectors import get_model, detect_centrioles

font = {'family': 'Helvetica',
        'weight': 'light',
        'size': 6}
matplotlib.rc('font', **font)


def metrics(field: Field,
            channel: int,
            annotation: np.ndarray,
            predictions: np.ndarray,
            tolerance: int) -> dict:
    """
    Compute the accuracy of the prediction on one field.
    :param field:
    :param channel:
    :param annotation:
    :param predictions:
    :param tolerance:
    :return: dictionary of fields
    """
    res = points_matching(annotation[:, [1, 0]],
                          predictions,
                          cutoff_distance=tolerance)
    perf = {
        'dataset': field.dataset.path.name,
        'field': field.name,
        'channel': channel,
        'n_actual': len(annotation),
        'n_preds': len(predictions),
        'tolerance': tolerance,
        'precision': res.precision.round(3),
        'recall': res.recall.round(3),
        'f1': res.f1.round(3),
    }
    return perf


def run_evaluation(dataset: Dataset, model, tolerances: list[int]) -> list:
    test_files = dataset.splits_for('test')

    perfs = []
    for field_name, channel in test_files:
        field = Field(field_name, dataset)
        annotation = field.annotation(channel)
        predictions = detect_centrioles(field, channel, model)
        for tol in tolerances:
            perf = metrics(field, channel, annotation, predictions, tol)
            perfs.append(perf)
    return perfs


def main():
    model_name = '2022-09-05_09:19:37'
    model = get_model(f'models/dev/{model_name}')
    tolerances = list(range(6))

    perfs = []
    for dataset_name in datasets:
        dataset = Dataset(PREFIX_REMOTE / dataset_name)
        performance = run_evaluation(dataset, model, tolerances)
        perfs.append(performance)

    performances_df = pd.DataFrame([s
                                    for p in perfs
                                    for s in p])

    path_out = Path('out')
    performances_df = performances_df.set_index('field')
    performances_df.to_csv(path_out / f'performances_{model_name}.csv')

    performances_df_3px = performances_df.loc[performances_df["tolerance"] == 3]
    performances_df_3px.to_csv(path_out / f'performances_{model_name}_3px.csv')

    summary = performances_df_3px.groupby(['dataset', 'channel']).agg('mean', 'std')['f1']
    summary.to_csv(path_out / f'performances_{model_name}_3px_summary.csv')


if __name__ == '__main__':
    main()
