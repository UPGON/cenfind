from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from stardist.models import StarDist2D

from centrack.data.base import Projection, Channel, Dataset, Field
from centrack.visualisation.outline import (
    Centre,
    Contour
)


def score_fov(dataset: Dataset, field: Field,
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

    projection = Projection(dataset, field)

    nuclei = Channel(projection, nuclei_channel)
    centres, nuclei = nuclei.extract_nuclei(model_nuclei)

    centrioles = Channel(projection, channel)
    foci = centrioles.detect_centrioles(model=model_foci)

    assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

    scored = []
    for pair in assigned:
        n, foci = pair
        scored.append({'fov': projection.name,
                       'channel': channel,
                       'nucleus': n.centre.position,
                       'score': len(foci),
                       'is_full': full_in_field(n.centre.position, .05, centrioles.data)
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
