import logging
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from stardist.models import StarDist2D

from centrack.core.data import Field, Dataset
from centrack.core.outline import Contour, Centre
from centrack.core.detectors import detect_centrioles
from spotipy.utils import points_matching


def prob2img(data):
    return (((2 ** 16) - 1) * data).astype('uint16')


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = tuple(int(c) for c in keypoint.pt)
    return res


def _resize_image(data):
    height, width = data.shape
    shrinkage_factor = int(height // 256)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(data,
                              dsize=(height_scaled, width_scaled),
                              fx=1, fy=1,
                              interpolation=cv2.INTER_NEAREST)
    return data_resized


def score_fov(field: Field,
              model_nuclei: StarDist2D, model_foci: Path,
              nuclei_channel: int, channel: int):
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
    :param dataset:
    :return:
    """

    nuclei = field.channel(nuclei_channel)
    centres, nuclei = nuclei.extract_nuclei(model_nuclei)

    centrioles = field.channel(channel)
    foci = centrioles.detect_centrioles(model=model_foci)
    foci = [
        Centre((y, x), f_id, 'Centriole',
               confidence=foci[0][x, y].round(3))
        for
        f_id, (x, y) in enumerate(foci[1])]

    assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

    scored = []
    for pair in assigned:
        n, foci = pair
        scored.append({'fov': field.name,
                       'channel': channel,
                       'nucleus': n.centre.position,
                       'score': len(foci),
                       'is_full': full_in_field(n.centre.position, .05, centrioles.projection)
                       })
    return scored


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""
    result = cv2.pointPolygonTest(nucleus.contour,
                                  focus.centre,
                                  measureDist=True)
    return result


def assign(foci: list, nuclei: list, vicinity: int) -> list[
    tuple[Any, list[Any]]]:
    """
    Assign detected centrioles to the nearest nucleus
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


def frac(x):
    return x.sum() / len(x)


def full_in_field(coordinate, fraction, image) -> bool:
    h, w = image.shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    if all([pad_lower < c < pad_upper for c in coordinate]):
        return True
    return False


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


def run_evaluation(dataset: Dataset, test_only, model, tolerances: list[int]) -> list:
    if test_only:
        fields = dataset.splits_for('test')
    else:
        fields_test = dataset.splits_for('test')
        fields_train = dataset.splits_for('train')
        fields = fields_train + fields_test

    perfs = []
    for field_name, channel in fields:
        field = Field(field_name, dataset)
        annotation = field.annotation(channel)
        predictions = detect_centrioles(field, channel, model)

        for tol in tolerances:
            perf = metrics(field, channel, annotation, predictions, tol)
            perfs.append(perf)
    return perfs


def score_summary(df):
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


def foci_prediction_prepare(foci, centriole_channel):
    foci_df = pd.DataFrame(foci)
    foci_df['channel'] = centriole_channel
    foci_df[['row', 'col']] = pd.DataFrame(foci_df['position'].to_list(),
                                           index=foci_df.index)
    foci_df = foci_df[['idx', 'channel', 'label', 'row', 'col', 'confidence']]
    result = foci_df.set_index('idx')

    return result
