import contextlib
import functools
import os
from pathlib import Path
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

from centrack.data.base import Dataset, Field
from centrack.experiments.constants import datasets, PREFIX_REMOTE
from centrack.scoring.measure import _resize_image, blob2point
from centrack.visualisation.outline import Centre, Contour, draw_foci


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


def detect_centrioles(data: Field, channel: int, model, prob_threshold=.5, min_distance=2) -> np.ndarray:
    data = data.channel(channel)
    data = normalize_fast2d(data)
    model = get_model(model)
    mask_preds, points_preds = model.predict(data,
                                             prob_thresh=prob_threshold,
                                             min_distance=min_distance)
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


def extract_nuclei(data, model: StarDist2D = None, annotation=None) -> Tuple[List[Centre], List[Contour]]:
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
        data_resized = _resize_image(data)
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


def main():
    methods = [detect_centrioles, sankaran, log_skimage, simpleblob_cv2]
    model_paths = {
        'sankaran': 'models/sankaran/dev/2022-09-05_09:23:45',
        'spotnet': 'models/dev/2022-09-02_14:31:28',
        'log_skimage': None,
        'simpleblob_cv2': None,
    }
    perfs = []
    for ds_name in datasets:
        ds = Dataset(PREFIX_REMOTE / ds_name)
        test_fields = ds.splits_for('test')
        for field_name, channel in test_fields:
            field = Field(field_name, ds)
            vis = field.channel(channel)
            annotation = field.annotation(channel)

            for method in methods:
                model_path = model_paths[method.__name__]
                foci, f1 = run_detection(method, field, annotation=annotation, channel=channel,
                                         model_path=model_path, tolerance=3)
                print(f"{field_name} using {method.__name__}: F1={f1}")

                perf = {'field': field.name,
                        'channel': channel,
                        'method': method.__name__,
                        'f1': f1}
                perfs.append(perf)

                mask = draw_foci(vis, foci)
                cv2.imwrite(f'out/images/{field.name}_max_C{channel}_preds_{method.__name__}.png', mask)

    pd.DataFrame(perfs).to_csv(f'out/perfs_blobdetectors.csv')


if __name__ == '__main__':
    main()
