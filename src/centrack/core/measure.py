import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from spotipy.model import SpotNet
from spotipy.utils import points_matching
from stardist.models import StarDist2D

from centrack.core.data import Dataset, Field
from centrack.core.detectors import detect_centrioles, extract_nuclei
from centrack.core.helpers import signed_distance, full_in_field
from centrack.core.outline import Centre


def assign(foci: list, nuclei: list, vicinity: int) -> list[tuple[Any, list[Any]]]:
    """
    Assign centrioles to nuclei in one field
    :param foci
    :param nuclei
    :param vicinity: the distance in pixels, below which centrioles are assigned
     to nucleus
    :return: List[Tuple[Centre, Contour]]
    """
    pairs = []
    _nuclei = nuclei.copy()
    while _nuclei:
        n = _nuclei.pop()
        assigned = []
        for f in foci:
            distance = signed_distance(f, n)
            if distance > vicinity:
                assigned.append(f)
        pairs.append((n, assigned))

    return pairs


def field_metrics(field: Field,
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
    if all((len(predictions), len(annotation))) > 0:
        res = points_matching(annotation[:, [1, 0]],
                              predictions,
                              cutoff_distance=tolerance)
    else:
        logging.warning('detected: %d; annotated: %d... Set precision and accuracy to zero' % (
            len(predictions), len(predictions)))
        res = SimpleNamespace()
        res.precision = 0.
        res.recall = 0.
    perf = {
        'dataset': field.dataset.path.name,
        'field': field.name,
        'channel': channel,
        'n_actual': len(annotation),
        'n_preds': len(predictions),
        'tolerance': tolerance,
        'precision': np.round(res.precision, 3),
        'recall': np.round(res.recall, 3),
        'f1': res.f1.round(3),
    }
    return perf


def dataset_metrics(dataset: Dataset, split, model, tolerance) -> list:
    fields = dataset.pairs(split)
    perfs = []
    for field_name, channel in fields:
        field = Field(field_name, dataset)
        annotation = field.annotation(channel)
        predictions = detect_centrioles(field, channel, model)
        perf = field_metrics(field, channel, annotation, predictions, tolerance)
        perfs.append(perf)
    return perfs


def field_score(field: Field,
                model_nuclei: StarDist2D,
                model_foci: SpotNet,
                nuclei_channel: int,
                channel: int) -> list:
    """
    1. Detect foci in the given channels
    2. Detect nuclei
    3. Assign foci to nuclei
    Return: dictionary of the record
    :param channel:
    :param nuclei_channel:
    :param model_foci:
    :param model_nuclei:
    :param field:
    :return:
    """

    centres, nuclei = extract_nuclei(field, nuclei_channel, model_nuclei)
    foci = detect_centrioles(data=field, channel=channel, model=model_foci)
    foci = [Centre((y, x), f_id, 'Centriole') for f_id, (x, y) in enumerate(foci)]

    assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

    scores = []
    for pair in assigned:
        n, foci = pair
        scores.append({'fov': field.name,
                       'channel': channel,
                       'nucleus': n.centre.position,
                       'score': len(foci),
                       'is_full': full_in_field(n.centre.position, field.projection, .05)
                       })
    return scores


def field_score_frequency(df):
    """
    Count the absolute frequency of number of centriole per image
    :param df: Df containing the number of centriole per nuclei
    :return: Df with absolut frequencies.
    """
    cuts = [0, 1, 2, 3, 4, 5, np.inf]
    labels = '0 1 2 3 4 +'.split(' ')

    df = df.set_index(['fov', 'channel'])
    result = pd.cut(df['score'], cuts, right=False,
                    labels=labels, include_lowest=True)

    result = (result
              .groupby(['fov', 'channel'])
              .value_counts()
              .sort_index()
              .reset_index())

    result = (result.rename({'level_2': 'score_cat',
                             'score': 'freq_abs'}, axis=1)
              .pivot(index=['fov', 'channel'], columns='score_cat'))
    return result
