from typing import Tuple

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from spotipy.utils import points_matching

from cenfind.core.data import Field
from cenfind.core.outline import Centre

def blob2point(keypoint: cv2.KeyPoint) -> tuple[int, ...]:
    res = (int(keypoint.pt[1]), int(keypoint.pt[0]))
    return res


def log_skimage(data: Field, channel: int, **kwargs) -> list:
    data = data.channel(channel)
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=1, max_sigma=2, num_sigma=10, threshold=.1)
    res = [(int(r), int(c)) for r, c, _ in foci]

    return res


def simpleblob_cv2(data: Field, channel: int, **kwargs) -> list:
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


def run_detection(method, data: Field,
                  annotation: np.ndarray,
                  tolerance,
                  channel=None,
                  model_path=None) -> Tuple[list[Centre], float]:
    _foci = method(data, foci_model_file=model_path, channel=channel)
    if type(_foci) == tuple:
        prob_map, _foci = _foci
    res = points_matching(annotation, _foci, cutoff_distance=tolerance)
    f1 = np.round(res.f1, 3)
    foci = [Centre((r, c), label='Centriole') for r, c in _foci]
    return foci, f1
