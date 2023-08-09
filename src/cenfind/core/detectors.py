import contextlib
import functools
import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from csbdeep.utils import normalize
from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters.thresholding import threshold_otsu
from spotipy.model import SpotNet
from spotipy.utils import normalize_fast2d
from stardist.models import StarDist2D

from cenfind.core.data import Field
from cenfind.core.log import get_logger
from cenfind.core.structures import Centriole, Nucleus
from cenfind.core.visualisation import draw_foci, resize_image

np.random.seed(1)
tf.random.set_seed(2)
tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = get_logger(__name__)


def extract_foci(field: Field, channel: int, foci_model_file: Path, prob_threshold=0.5, min_distance=2, ) -> List[
    Centriole]:
    """
    Detect centrioles in Field as row, col, row major
    :param field:
    :param foci_model_file:
    :param channel:
    :param prob_threshold:
    :param min_distance:
    :return:
    """
    logger.info("Processing %s / %d" % (field.name, channel))
    data = field.data[channel, ...]

    @functools.lru_cache(maxsize=None)
    def get_model(model):
        path = Path(model)
        if not path.is_dir():
            raise (FileNotFoundError(f"{path} is not a directory"))

        return SpotNet(None, name=path.name, basedir=str(path.parent))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        model = get_model(foci_model_file)
        _, points_preds = model.predict(
            data, prob_thresh=prob_threshold, min_distance=min_distance, verbose=False
        )

    foci = [
        Centriole(field=field, channel=channel, centre=(r, c), index=f_id, label='Centriole') for f_id, (r, c) in
        enumerate(points_preds.tolist())
    ]

    centrosomes_mask = np.zeros(data.shape, dtype="uint8")
    centrosomes_mask = draw_foci(centrosomes_mask, foci, radius=min_distance * 2)

    centrosomes_map = measure.label(centrosomes_mask)
    centrosomes_centroids = measure.regionprops(centrosomes_map)

    for f in foci:
        foci_index = centrosomes_map[f.centre]
        centrosome_centroid = centrosomes_centroids[foci_index - 1].centroid
        f.parent = Centriole(field=field, channel=channel, centre=centrosome_centroid, label="Centrosome")

    if len(foci) == 0:
        logger.warning("No centrioles (channel: %s) has been detected in %s" % (channel, field.name))

    logger.info("(%s), channel %s: foci: %s" % (field.name, channel, len(foci)))
    return foci


def extract_nuclei(field: Field, channel: int, model: StarDist2D = None) -> List[Nucleus]:
    """
    Extract the nuclei from the field.
    :param field:
    :param channel:
    :param model:

    :return: List of Contours.

    """
    if model is None:
        from stardist.models import StarDist2D
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            model = StarDist2D.from_pretrained("2D_versatile_fluo")

    data = field.data[channel, ...]

    data_resized = resize_image(data)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        labels, _ = model.predict_instances(normalize(data_resized))
    labels = cv2.resize(
        labels, dsize=data.shape, fx=1, fy=1, interpolation=cv2.INTER_NEAREST
    )

    if len(labels) == 0:
        logger.warning("No nucleus has been detected in %s" % field.name)
        return []
    labels_id = np.unique(labels)

    nuclei = []
    for nucleus_index, nucleus_label in enumerate(labels_id):
        if nucleus_label == 0:
            continue
        sub_mask = np.zeros_like(labels, dtype="uint8")
        sub_mask[labels == nucleus_label] = 1
        contour, _ = cv2.findContours(
            sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        nucleus = Nucleus(field=field, channel=channel, contour=contour[0], label="Nucleus", index=nucleus_index - 1)
        nuclei.append(nucleus)

    logger.info("(%s), channel %s: foci: %s" % (field.name, channel, len(nuclei)))

    return nuclei


def extract_cilia(field: Field, channel, sigma=5.0, eccentricity=.9, area=200) -> List[Centriole]:
    data = field.data[channel, ...]
    resc = rescale_intensity(data, out_range='uint8')

    h_elems = hessian_matrix(resc, sigma=sigma, order='rc')
    _, minima_ridges = hessian_matrix_eigvals(h_elems)
    threshold = threshold_otsu(minima_ridges)

    mask = minima_ridges < threshold
    labels = measure.label(mask)
    props = measure.regionprops(labels, mask)

    if not props:
        logger.warning("No cilium (channel: %s) has been detected in %s" % (channel, field.name))

    result = []
    for prop in props:
        if prop.eccentricity > eccentricity and prop.area > area:
            r, c = prop.centroid
            result.append(Centriole(field=field, channel=channel, centre=(int(r), int(c)), label='Cilium'))

    return result
