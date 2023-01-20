import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from spotipy.utils import points_matching
from stardist.models import StarDist2D

from cenfind.core.data import Dataset, Field
from cenfind.core.detectors import extract_foci, extract_nuclei
from cenfind.core.outline import Centre, draw_foci, Contour

import cv2


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""

    result = cv2.pointPolygonTest(nucleus.contour,
                                  focus.to_cv2(),
                                  measureDist=True)
    return result


def full_in_field(coordinate, image_shape, fraction) -> bool:
    h, w = image_shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    if all([pad_lower < c < pad_upper for c in coordinate]):
        return True
    return False


def flag(is_full: bool) -> tuple:
    return (0, 255, 0) if is_full else (0, 0, 255)


def infer_centrosomes(foci: list, img_shape: tuple, distance=.6) -> list:
    """
    Add centrosome to a list of foci.
    Args:
        foci:
        img_shape:
        distance: twice the inter-centriole distance in micrometres (2 x .3 um)

    Returns:

    """
    _foci = foci.copy()

    centrosomes_mask = np.zeros(img_shape, dtype='uint8')
    centrosomes_mask = draw_foci(centrosomes_mask, _foci, radius=distance)

    centrosomes_map = label(centrosomes_mask)
    centrosomes_centroids = regionprops(centrosomes_map)

    for f in _foci:
        foci_index = centrosomes_map[f.centre]
        centrosome_centroid = centrosomes_centroids[foci_index - 1].centroid
        centrosome_centroid = tuple(int(c) for c in centrosome_centroid)
        f.parent = Centre(centrosome_centroid, label='Centrosome')

    return foci


def assign(foci: list, nuclei: list, vicinity: float, pixel_size: float) -> list[tuple[Any, list[Any]]]:
    """
    Assign centrioles to nuclei in one field
    :param foci
    :param nuclei
    :param vicinity: the distance in pixels, below which centrioles are assigned
     to nucleus
    :param pixel_size: in micrometres
    :return: List[Tuple[Centre, Contour]]

    """
    if len(nuclei) == 0:
        raise ValueError('foci is an empty list')
    _foci = foci.copy()
    _nuclei = nuclei.copy()

    vicinity_pixel = int(vicinity / pixel_size)

    nuclei_pos = [tuple(n.centre.to_numpy()) for n in _nuclei]

    # Initialise the pairs so that nuclei
    # with no centrioles are maintained in the output
    container = defaultdict(list)
    for k in nuclei_pos:
        container.setdefault(k, [])

    # Take each focus and compute the distance to all nuclei
    # and assign it to the nearest nucleus, if it is in the vicinity
    while len(_foci):
        f = _foci.pop()
        centrosome = f.parent
        dists = []
        for n in _nuclei:
            nuclei_pos = tuple(n.centre.to_numpy())
            dist = signed_distance(centrosome, n)
            dists.append((nuclei_pos, dist))
        nuclei_pos_nearest = max(dists, key=lambda t: t[1])
        nearest_nucleus, max_signed_dist = nuclei_pos_nearest
        if max_signed_dist > vicinity_pixel:
            container[nearest_nucleus].append(f)
            continue

    pairs = [(k, v) for k, v in container.items()]

    return pairs


def field_score(field: Field,
                model_nuclei: StarDist2D,
                model_foci: Path,
                nuclei_channel: int,
                factor,
                vicinity,
                channel: int) -> Tuple[np.ndarray, list]:
    """
    1. Detect foci in the given channels
    2. Detect nuclei
    3. Assign foci to nuclei
    :param channel:
    :param nuclei_channel:
    :param model_foci:
    :param model_nuclei:
    :param field:
    :return: list(foci, nuclei, assigned, scores)
    """

    image_shape = field.projection.shape[1:]
    centres, nuclei = extract_nuclei(field, nuclei_channel, factor, model_nuclei)
    if len(nuclei) == 0:
        raise ValueError('No nucleus has been detected')
    prob_map, foci = extract_foci(data=field, foci_model_file=model_foci, channel=channel)
    foci = [Centre((r, c), f_id, 'Centriole') for f_id, (r, c) in enumerate(foci)]

    foci = infer_centrosomes(foci, image_shape, distance=.6)
    assigned = assign(foci=foci, nuclei=nuclei, vicinity=vicinity, pixel_size=.1025)

    scores = []
    for pair in assigned:
        nucleus, focus = pair
        scores.append({'fov': field.name,
                       'channel': channel,
                       'nucleus': nucleus,
                       'score': len(focus),
                       'is_full': full_in_field(nucleus, image_shape, .05)
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


def field_score_frequency(df, by='field'):
    """
    Count the absolute frequency of number of centriole per well or per field
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
    result = result.rename({'level_2': 'score_cat'}, axis=1)
    
    if by == 'well':
        result[['well', 'field']] = result['fov'].str.split('_', expand=True)
        result = result.groupby(['well', 'channel', 'score_cat']).sum()
        result = result.reset_index()
        result = result.pivot(index=['well', 'channel'], columns='score_cat')
        result.reset_index().sort_values(['channel', 'well'])
    else:
        result = result.groupby(['fov', 'channel', 'score_cat']).sum()
        result = result.reset_index()
        result = result.pivot(index=['fov', 'channel'], columns='score_cat')
        result.reset_index().sort_values(['channel', 'fov'])
    
    return result


def save_foci(foci_list: list[Centre], dst: str, logger=None) -> None:
    if len(foci_list) == 0:
        array = np.array([])
        if logger is not None:
            logger.info('No centriole detected')
        else:
            print('No centriole detected')
    else:
        array = np.asarray(np.stack([c.to_numpy() for c in foci_list]))
        array = array[:, [1, 0]]
    np.savetxt(dst, array, delimiter=',', fmt='%u')
