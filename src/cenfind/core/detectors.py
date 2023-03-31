import contextlib
import functools
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from skimage.measure import label, regionprops
from spotipy.model import SpotNet
from spotipy.utils import normalize_fast2d

from cenfind.core.outline import draw_foci

np.random.seed(1)
tf.random.set_seed(2)
from cenfind.core.data import Field
from cenfind.core.outline import Centre

np.random.seed(1)
tf.random.set_seed(2)


@functools.lru_cache(maxsize=None)
def get_model(model):
    path = Path(model)
    if not path.is_dir():
        raise (FileNotFoundError(f"{path} is not a directory"))

    return SpotNet(None, name=path.name, basedir=str(path.parent))


def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = (int(keypoint.pt[1]), int(keypoint.pt[0]))
    return res


def extract_foci(
        field: Field,
        foci_model_file: Path,
        channel: int,
        prob_threshold=0.5,
        min_distance=2,
        **kwargs,
) -> List[Centre]:
    """
    Detect centrioles as row, col, row major
    :param field:
    :param foci_model_file:
    :param channel:
    :param prob_threshold:
    :param min_distance:
    :param kwargs:
    :return:
    """
    data = field.channel(channel)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        model = get_model(foci_model_file)
        _, points_preds = model.predict(
            data, prob_thresh=prob_threshold, min_distance=min_distance, verbose=False
        )
    foci = [
        Centre((r, c), f_id, "Centriole") for f_id, (r, c) in enumerate(points_preds)
    ]

    centrosomes_mask = np.zeros(data.shape, dtype="uint8")
    centrosomes_mask = draw_foci(centrosomes_mask, foci, radius=min_distance * 2)

    centrosomes_map = label(centrosomes_mask)
    centrosomes_centroids = regionprops(centrosomes_map)

    for f in foci:
        foci_index = centrosomes_map[f.centre]
        centrosome_centroid = centrosomes_centroids[foci_index - 1].centroid
        centrosome_centroid = tuple(int(c) for c in centrosome_centroid)
        f.parent = Centre(centrosome_centroid, label="Centrosome")

    return foci





def log_skimage(data: Field, channel: int, **kwargs) -> list:
    data = data.channel(channel)
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=1, max_sigma=2, num_sigma=10, threshold=0.1)
    res = [(int(r), int(c)) for r, c, _ in foci]

    return res


def simpleblob_cv2(data: Field, channel: int, **kwargs) -> list:
    data = data.channel(channel)
    foci = rescale_intensity(data, out_range="uint8")
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
