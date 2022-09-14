import contextlib
import functools
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from centrosome_analysis import centrosome_analysis_backend as cab
from csbdeep.utils import normalize
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from spotipy.model import SpotNet
from spotipy.utils import normalize_fast2d, points_matching
from stardist.models import StarDist2D

from centrack.core.data import Dataset
from centrack.core.data import Field
from centrack.core.outline import Centre, Contour


def frac(x):
    return x.sum() / len(x)


def full_in_field(coordinate, image, fraction) -> bool:
    _, h, w = image.shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    if all([pad_lower < c < pad_upper for c in coordinate]):
        return True
    return False


def flag(is_full: bool) -> tuple:
    return (0, 0, 255) if is_full else (0, 255, 0)


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


def score_fov(field: Field,
              model_nuclei: StarDist2D,
              model_foci: SpotNet,
              nuclei_channel: int,
              channel: int):
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

    centres, nuclei = extract_nuclei(field, nuclei_channel, model_nuclei)
    foci = detect_centrioles(data=field, channel=channel, model=model_foci)
    foci = [Centre((y, x), f_id, 'Centriole') for f_id, (x, y) in enumerate(foci)]

    assigned = assign(foci=foci, nuclei=nuclei, vicinity=-50)

    scored = []
    for pair in assigned:
        n, foci = pair
        scored.append({'fov': field.name,
                       'channel': channel,
                       'nucleus': n.centre.position,
                       'score': len(foci),
                       'is_full': full_in_field(n.centre.position, field.projection, .05)
                       })
    return scored


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


def resize_image(data):
    height, width = data.shape
    shrinkage_factor = int(height // 256)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(data,
                              dsize=(height_scaled, width_scaled),
                              fx=1, fy=1,
                              interpolation=cv2.INTER_NEAREST)
    return data_resized


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = tuple(int(c) for c in keypoint.pt)
    return res


@functools.lru_cache(maxsize=None)
def get_model(model):
    path = Path(model)
    if not path.is_dir():
        raise (FileNotFoundError(f"{path} is not a directory"))

    return SpotNet(None, name=path.name, basedir=str(path.parent))


def log_skimage(data: Field, channel: int) -> list:
    data = data.channel(channel)
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=.5, max_sigma=5, num_sigma=10, threshold=.1)
    res = [(int(c), int(r)) for r, c, _ in foci]

    return res


def simpleblob_cv2(data: Field, channel: int) -> list:
    data = data.channel(channel)
    foci = rescale_intensity(data, out_range='uint8')
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 100
    params.minDistBetweenBlobs = 1
    params.minThreshold = 0
    params.maxThreshold = 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(foci)

    res = [blob2point(kp) for kp in keypoints]

    return res


def detect_centrioles(data: Field, channel: int, model: SpotNet, prob_threshold=.5, min_distance=2) -> np.ndarray:
    data = data.channel(channel)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        mask_preds, points_preds = model.predict(data,
                                                 prob_thresh=prob_threshold,
                                                 min_distance=min_distance, verbose=False)
    return points_preds[:, [1, 0]]


def sankaran(data: Field, foci_model_file) -> np.ndarray:
    data = data.projection[1:, :, :]
    foci_model = cab.load_foci_model(foci_model_file=foci_model_file)
    foci, foci_scores = cab.run_detection_model(data, foci_model)
    detections = foci[foci_scores > .99, :]
    detections = np.round(detections).astype(int)
    return detections


def run_detection(method, data: Field,
                  annotation: np.ndarray,
                  tolerance,
                  channel=None,
                  model_path=None) -> Tuple[np.ndarray, float]:
    foci = method(data, foci_model_file=model_path, channel=channel)
    res = points_matching(annotation, foci, cutoff_distance=tolerance)
    f1 = np.round(res.f1, 3)
    return foci, f1


def extract_nuclei(field: Field, channel: int, model: StarDist2D = None, annotation=None) -> Tuple[
    List[Centre], List[Contour]]:
    """
    Extract the nuclei from the nuclei image
    :param data:
    :param annotation: a mask with pixels labelled for each centre
    :param model:
    :return: List of Contours.
    """

    if annotation is not None:
        nuclei_detected = annotation
    elif model is not None:
        data = field.channel(channel)
        data_resized = resize_image(data)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            labels, coords = model.predict_instances(normalize(data_resized))
        nuclei_detected = cv2.resize(labels, dsize=data.shape,
                                     fx=1, fy=1,
                                     interpolation=cv2.INTER_NEAREST)

    else:
        raise ValueError("Please provide either an annotation or a model")

    labels_id = np.unique(nuclei_detected)

    cnts = []
    for nucleus_id in labels_id:
        if nucleus_id == 0:
            continue
        sub_mask = np.zeros_like(nuclei_detected, dtype='uint8')
        sub_mask[nuclei_detected == nucleus_id] = 255
        contour, hierarchy = cv2.findContours(sub_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        cnts.append(contour[0])

    contours = tuple(cnts)
    contours = [Contour(c, 'Nucleus', c_id, confidence=-1) for c_id, c in
                enumerate(contours)]

    centres = [c.centre for c in contours]

    return centres, contours
