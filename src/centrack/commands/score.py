import argparse
import contextlib
import functools
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any

from cv2 import cv2
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from stardist.models import StarDist2D

from centrack.commands.outline import (
    Centre,
    Contour,
    prepare_background,
    draw_annotation
)
from centrack.commands.status import (
    DataSet,
    load_projection,
)
from spotipy.spotipy.model import SpotNet
from spotipy.spotipy.utils import normalize_fast2d

logger_score = logging.getLogger()
logger_score.setLevel(logging.INFO)


@functools.lru_cache(maxsize=None)
def get_model(model):
    path = Path(model)
    if not path.is_dir():
        raise (FileNotFoundError(f"{path} is not a directory"))

    return SpotNet(None, name=path.name, basedir=str(path.parent))


class Detector(ABC):
    def __init__(self, plane, organelle):
        self.plane = plane
        self.organelle = organelle

    @abstractmethod
    def _mask(self):
        pass

    @abstractmethod
    def detect(self):
        pass


class CentriolesDetector(Detector):
    """
    Combine a preprocessing and a detection step and return a list of centres.
    """

    def _mask(self):
        transformed = self.plane
        return transformed

    def detect(self, interpeak_min=3):
        current_path = Path(__file__).parent.parent.parent.parent
        path_to_model = current_path / 'models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300'
        model = get_model(
            model=path_to_model)
        image = self.plane
        x = normalize_fast2d(image)
        prob_thresh = .5

        foci = model.predict(x,
                             prob_thresh=prob_thresh,
                             show_tile_progress=False)

        return [Centre((y, x), f_id, self.organelle, confidence=-1) for
                f_id, (x, y) in enumerate(foci[1])]


class NucleiDetector(Detector):
    """
    Resize a DAPI image and run StarDist
    """

    def _mask(self):
        return cv2.resize(self.plane, dsize=(256, 256),
                          fx=1, fy=1,
                          interpolation=cv2.INTER_NEAREST)

    def detect(self):
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        transformed = self._mask()

        transformed = transformed
        labels, coords = model.predict_instances(normalize(transformed))

        nuclei_detected = cv2.resize(labels, dsize=(2048, 2048),
                                     fx=1, fy=1,
                                     interpolation=cv2.INTER_NEAREST)

        labels_id = np.unique(nuclei_detected)
        cnts = []
        for nucleus_id in labels_id:
            if nucleus_id == 0:
                continue
            submask = np.zeros_like(nuclei_detected, dtype='uint8')
            submask[nuclei_detected == nucleus_id] = 255
            contour, hierarchy = cv2.findContours(submask,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            cnts.append(contour[0])
        contours = tuple(cnts)
        return [Contour(c, self.organelle, c_id, confidence=-1) for c_id, c in
                enumerate(contours)]


def extract_centrioles(data):
    """
    Extract the centrioles from the channel image.
    :param data:
    :return: List of Points
    """
    focus_detector = CentriolesDetector(data, 'Centriole')
    return focus_detector.detect()


def extract_nuclei(data):
    """
    Extract the nuclei from the nuclei image.
    :param data:
    :return: List of Contours.
    """
    nuclei_detector = NucleiDetector(data, 'Nucleus')
    return nuclei_detector.detect()


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""
    result = cv2.pointPolygonTest(nucleus.contour,
                                  focus.centre,
                                  measureDist=True)
    return result


def assign(foci: list, nuclei: list, vicinity: int) -> list[tuple[Any, list[Any]]]:
    """
    Assign detected centrioles to the nearest nucleus.
    :param foci
    :param nuclei
    :param vicinity: the distance in pixels, below which centrioles are assigned to nucleus
    :return: List[Tuple[Centre, Contour]]
    """
    pairs = []
    _nuclei = nuclei.copy()
    while _nuclei:
        n = _nuclei.pop()
        assigned = []
        for f in foci:
            distance = signed_distance(f, n)
            if distance > -50:
                assigned.append(f)
        pairs.append((n, assigned))

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description='CENTRACK: Automatic centriole scoring')

    parser.add_argument('dataset',
                        type=Path,
                        help='path to the dataset')
    parser.add_argument('channel',
                        type=int,
                        help='channel position to use for foci detection, e.g., 1, 2 or 3')
    parser.add_argument('-t', '--test',
                        type=int,
                        help='test; only run on the ith image')

    return parser.parse_args()


def cli():
    args = parse_args()
    path_dataset = args.dataset
    dataset = DataSet(path_dataset)
    centriole_channel = args.channel
    fields = tuple(f for f in dataset.projections.glob('*.tif') if
                   not f.name.startswith('.'))

    logger_score.info('Starting Centrack...')
    logger_score.debug('Working at %s', path_dataset)
    logger_score.debug('%s files were found', len(fields))
    if args.test:
        logger_score.warning('Test mode enabled: only one field will be processed.')
        fields = [fields[args.test]]

    path_scores = path_dataset / 'results'
    path_scores.mkdir(exist_ok=True)

    pairs = []
    scored = []
    for path in fields:
        logger_score.info('Loading %s', path.name)

        data = load_projection(path)

        if len(data.shape) != 3:
            raise ValueError(f"File {path} has shape {data.shape}")

        foci = data[centriole_channel, :, :]
        nuclei = data[0, :, :]  # 0 is by default the position of the DAPI channel

        # This skips the print calls in spotipy
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            foci_detected = extract_centrioles(foci)
            nuclei_detected = extract_nuclei(nuclei)
        logger_score.info('%s: (%s foci, %s nuclei)', path.name, len(foci_detected),
                          len(nuclei_detected))

        assigned = assign(foci=foci_detected,
                          nuclei=nuclei_detected,
                          vicinity=-50)

        logger_score.debug('Creating annotation image...')
        background = prepare_background(nuclei, foci)
        annotation = draw_annotation(background, assigned, foci_detected,
                                     nuclei_detected)

        file_name = path.name.removesuffix(".tif")
        destination_path = path_scores / f'{file_name}_annot.png'
        successful = cv2.imwrite(str(destination_path), annotation)

        if successful:
            logger_score.debug('Saved at %s', destination_path)

        for pair in assigned:
            n, foci = pair
            scored.append({'fov': path.name,
                           'channel': centriole_channel,
                           'nucleus': n.centre.position,
                           'centrioles_n': len(foci),
                           })
            for focus in foci:
                pairs.append({'fov': path.name,
                              'channel': centriole_channel,
                              'nucleus': n.centre.position,
                              'centriole': focus.centre,
                              })

    results = pd.DataFrame(pairs)
    results.to_csv(path_scores / 'centrioles.csv')
    logger_score.info('Results saved at %s', str(path_scores / 'centrioles.csv'))

    scores = pd.DataFrame(scored)
    scores.to_csv(path_scores / 'score_primary.csv')
    logger_score.info('Results saved at %s', str(path_scores / 'score_primary.csv'))


if __name__ == '__main__':
    cli()
