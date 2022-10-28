import logging
from types import SimpleNamespace
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from spotipy.utils import points_matching
from stardist.models import StarDist2D

from cenfind.core.data import Dataset, Field
from cenfind.core.detectors import extract_foci, extract_nuclei
from cenfind.core.helpers import signed_distance, full_in_field
from cenfind.core.outline import Centre


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

def field_score(field: Field,
                model_nuclei: StarDist2D,
                model_foci: Path,
                nuclei_channel: int,
                channel: int) -> (np.ndarray, list):
    """
    1. Detect foci in the given channels
    2. Detect nuclei
    3. Assign foci to nuclei
    :param channel:
    :param nuclei_channel:
    :param model_foci:
    :param model_nuclei:
    :param field:
    :return: list(foci, nuclei, scores)
    """

    centres, nuclei = extract_nuclei(field, nuclei_channel, model_nuclei)
    prob_map, foci = extract_foci(data=field, foci_model_file=model_foci, channel=channel)
    foci = [Centre((r, c), f_id, 'Centriole') for f_id, (r, c) in enumerate(foci)]

    assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

    scores = []
    for pair in assigned:
        n, _foci = pair
        scores.append({'fov': field.name,
                       'channel': channel,
                       'nucleus': n.centre.position,
                       'score': len(_foci),
                       'is_full': full_in_field(n.centre.position, field.projection, .05)
                       })
    return foci, nuclei, assigned, scores

def field_metrics(field: Field,
                  channel: int,
                  annotation: np.ndarray,
                  predictions: np.ndarray,
                  tolerance: int,
                  threshold: float) -> dict:
    """
    Compute the accuracy of the prediction on one field.
    :param threshold:
    :param field:
    :param channel:
    :param annotation:
    :param predictions:
    :param tolerance:
    :return: dictionary of fields
    """
    if all((len(predictions), len(annotation))) > 0:
        res = points_matching(annotation,
                              predictions,
                              cutoff_distance=tolerance)
    else:
        logging.warning('threshold: %f; detected: %d; annotated: %d... Set precision and accuracy to zero' % (
            threshold, len(predictions), len(annotation)))
        res = SimpleNamespace()
        res.precision = 0.
        res.recall = 0.
        res.f1 = 0.
        res.tp = 0,
        res.fp = 0,
        res.fn = 0
    perf = {
        'dataset': field.dataset.path.name,
        'field': field.name,
        'channel': channel,
        'n_actual': len(annotation),
        'n_preds': len(predictions),
        'threshold': threshold,
        'tolerance': tolerance,
        'tp': res.tp[0] if type(res.tp) == tuple else res.tp,
        'fp': res.fp[0] if type(res.fp) == tuple else res.fp,
        'fn': res.fn[0] if type(res.fn) == tuple else res.fn,
        'precision': np.round(res.precision, 3),
        'recall': np.round(res.recall, 3),
        'f1': np.round(res.f1, 3),
    }
    return perf


def dataset_metrics(dataset: Dataset, split: str, model: Path, tolerance, threshold) -> tuple[dict, list]:
    perfs = []
    prob_maps = {}
    for field, channel in dataset.pairs(split):
        annotation = field.annotation(channel)
        prob_map, predictions = extract_foci(field, model, channel, prob_threshold=threshold)
        perf = field_metrics(field, channel, annotation, predictions, tolerance, threshold=threshold)
        perfs.append(perf)
        prob_maps[field.name] = prob_map
    return prob_maps, perfs





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
              )
    result.name = 'freq_abs'
    result = (result.sort_index()
              .reset_index()
              )
    result = result.rename({'score': 'score_cat'}, axis=1)

    result = result.pivot(index=['fov', 'channel'], columns='score_cat')
    return result
