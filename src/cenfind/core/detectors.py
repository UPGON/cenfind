import contextlib
import os
from pathlib import Path
from typing import Tuple, List

import torch
import cv2
import numpy as np
from centrosome_analysis import centrosome_analysis_backend as cab
from csbdeep.utils import normalize
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log
from spotipy.model import SpotNet
from spotipy.utils import normalize_fast2d
from spotipy.utils import points_matching
from stardist.models import StarDist2D

from cenfind.core.data import Field
from cenfind.core.helpers import blob2point, get_model
from cenfind.core.helpers import resize_image
from cenfind.core.outline import Centre, Contour


def log_skimage(data: Field, channel: int, **kwargs) -> list:
    data = data.channel(channel)
    data = rescale_intensity(data, out_range=(0, 1))
    foci = blob_log(data, min_sigma=.5, max_sigma=5, num_sigma=10, threshold=.1)
    res = [(int(c), int(r)) for r, c, _ in foci]

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


def spotnet(data: Field,
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
    model = get_model(foci_model_file)
    data = data.channel(channel)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        data = normalize_fast2d(data)
        mask_preds, points_preds = model.predict(data,
                                                 prob_thresh=prob_threshold,
                                                 min_distance=min_distance, verbose=False)
    return mask_preds, points_preds


def sankaran(data: Field, foci_model_file, **kwargs) -> np.ndarray:
    data = data.projection[1:, :, :]
    with torch.no_grad():
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
    if type(foci) == tuple:
        prob_map, foci = foci
    res = points_matching(annotation, foci, cutoff_distance=tolerance)
    f1 = np.round(res.f1, 3)
    return foci, f1


def extract_nuclei(field: Field, channel: int, model: StarDist2D = None, annotation=None) -> Tuple[
    List[Centre], List[Contour]]:
    """
    Extract the nuclei from the nuclei image
    :param channel:
    :param field:
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
