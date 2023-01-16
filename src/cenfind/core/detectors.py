import contextlib
import os
from pathlib import Path
from typing import Tuple, List
import functools

import cv2
import numpy as np
from csbdeep.utils import normalize
from spotipy.utils import normalize_fast2d
from spotipy.model import SpotNet
from stardist.models import StarDist2D

from cenfind.core.data import Field
from cenfind.core.outline import resize_image
from cenfind.core.outline import Centre, Contour


@functools.lru_cache(maxsize=None)
def get_model(model):
    path = Path(model)
    if not path.is_dir():
        raise (FileNotFoundError(f"{path} is not a directory"))

    return SpotNet(None, name=path.name, basedir=str(path.parent))


def extract_foci(data: Field,
                 foci_model_file: Path,
                 channel: int,
                 prob_threshold=.5,
                 min_distance=2, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect centrioles as row, col, row major
    :param data:
    :param foci_model_file:
    :param channel:
    :param prob_threshold:
    :param min_distance:
    :param kwargs:
    :return:
    """
    data = data.channel(channel)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        model = get_model(foci_model_file)
        mask_preds, points_preds = model.predict(data,
                                                 prob_thresh=prob_threshold,
                                                 min_distance=min_distance, verbose=False)
    return mask_preds, points_preds


def extract_nuclei(field: Field,
                   channel: int,
                   factor: int,
                   model: StarDist2D = None,
                   annotation=None) -> Tuple[
    List[Centre], List[Contour]]:
    """
    Extract the nuclei from the nuclei image
    :param field:
    :param channel:
    :param factor: the factor related to pixel size
    :param model:
    :param annotation: a mask with pixels labelled for each centre

    :return: List of Contours.

    """

    if annotation is not None:
        nuclei_detected = annotation
    elif model is not None:
        data = field.channel(channel)
        data_resized = resize_image(data, factor)
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
