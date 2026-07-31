import contextlib
import functools
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
from cenfind.core.structures import Centriole, Nucleus
from cenfind.core.visualisation import draw_foci, resize_image

tf.get_logger().setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _load_foci_model(model_dir: str) -> SpotNet:
    """
    Loads (and caches) a SpotNet model from a directory.

    Defined at module scope so the lru_cache persists across calls to
    extract_foci: previously this was redefined inside extract_foci on every
    call, which meant the model was reloaded from disk for every field/channel
    instead of once per process.
    """
    path = Path(model_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        return SpotNet(None, name=path.name, basedir=str(path.parent))


def extract_foci(field: Field, channel: int, foci_model_file: Path,
                 prob_threshold=0.5, min_distance=2, ) -> List[Centriole]:
    """
    Detects centrioles in Field as (row, col) position.

    :param field: Field of view to search for centrioles.
    :param foci_model_file: SpotNet trained model file.
    :param channel: Channel to use in the field.
    :param prob_threshold: Probability threshold used for the cutoff (default: 0.5).
    :param min_distance: Minimal distance between two centrioles (default: 2 pixels).
    :return: List of centriole objects.
    """
    logger.info("Processing %s / %d" % (field.name, channel))
    if field.data.ndim != 3:
        raise ValueError("Bad data shape: %s; Ensure that the image is CXY" % field.data.shape)
    data = field.data[channel, ...]

    model = _load_foci_model(str(foci_model_file))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        _, points_preds = model.predict(
            data, prob_thresh=prob_threshold, min_distance=min_distance, verbose=False
        )

    foci = []
    for f_id, (r, c) in enumerate(points_preds.tolist()):
        foci.append(Centriole(field=field, channel=channel, centre=(r, c), index=f_id, label="Centriole"))

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


@functools.lru_cache(maxsize=None)
def _load_nuclei_model() -> StarDist2D:
    """
    Loads (and caches) the pretrained StarDist nuclei-segmentation model.

    Cached at module scope for the same reason as _load_foci_model: without
    it, every extract_nuclei(field, channel) call with no explicit model
    (as in the score CLI loop) reloads the pretrained weights from disk.
    """
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        return StarDist2D.from_pretrained("2D_versatile_fluo")


def extract_nuclei(field: Field, channel: int, model: StarDist2D = None) -> List[Nucleus]:
    """
    Extracts the nuclei from the field.

    :param field: Field of view to search for nuclei.
    :param channel: Channel to use in the field.
    :param model: Model instance of StarDist.

    :return: List of Nuclei.

    """
    if model is None:
        model = _load_nuclei_model()

    if field.data.ndim == 2:
        data = field.data
    elif field.data.ndim == 3:
        data = field.data[channel, ...]
    else:
        raise ValueError("Bad data shape: %s; Ensure that the image is CXY" % field.data.shape)

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

    logger.info("Nuclei extraction (channel %s) in %s: %s" % (channel, field.name, len(nuclei)))

    return nuclei


def extract_cilia(field: Field, channel, sigma=5.0, eccentricity=.9, area=200) -> List[Centriole]:
    """
    Extracts the cilia using the Hessian.

    :param field: Field of view to search for nuclei.
    :param channel: Channel to use in the field.
    :param sigma: Scale for the blob detection.
    :param eccentricity: Sphericity of the blobs to be detected.
    :param area: Filter area to use.
    :return: List of cilia Point objects.
    """
    data = field.data[channel, ...]
    resc = rescale_intensity(data, out_range="uint8")

    h_elems = hessian_matrix(resc, sigma=sigma, order="rc")
    _, minima_ridges = hessian_matrix_eigvals(h_elems)
    threshold = threshold_otsu(minima_ridges)

    mask = minima_ridges < threshold
    labels = measure.label(mask)
    props = measure.regionprops(labels, mask)

    result = []
    for prop in props:
        if prop.eccentricity > eccentricity and prop.area > area:
            r, c = prop.centroid
            result.append(Centriole(field=field, channel=channel, centre=(int(r), int(c)), label="Cilium"))

    logger.info("Cilium extraction (channel %s) in %s: %s" % (channel, field.name, len(result)))

    return result
