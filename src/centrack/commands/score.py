import argparse
import contextlib
import functools
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from cv2 import cv2
from stardist.models import StarDist2D

from centrack.commands.outline import (
    Centre,
    Contour, prepare_background, draw_annotation
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
    def __init__(self, data, channel, organelle):
        self.data = data
        self.channel = channel
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
        transformed = self.data[self.channel, :, :]
        return transformed

    def detect(self, interpeak_min=3):
        current_path = Path(__file__).parent.parent.parent.parent
        path_to_model = current_path / 'models/leo3_multiscale_True_mae_aug_1_sigma_1.5_split_2_batch_2_n_300'
        model = get_model(
            model=path_to_model)
        image = self._mask()
        x = normalize_fast2d(image)
        prob_thresh = .5

        foci = model.predict(x,
                             prob_thresh=prob_thresh,
                             show_tile_progress=False)

        return [
            Centre((y, x), f_id, self.organelle,
                   confidence=foci[0][x, y].round(3))
            for
            f_id, (x, y) in enumerate(foci[1])]


class NucleiDetector(Detector):
    """
    Resize a DAPI image and run StarDist
    """

    def _mask(self):
        _, height, width = self.data.shape
        shrinkage_factor = 8
        height_scaled = int(height // shrinkage_factor)
        width_scaled = int(width // shrinkage_factor)
        return shrinkage_factor, cv2.resize(self.data[self.channel, :, :],
                                            dsize=(height_scaled, width_scaled),
                                            fx=1, fy=1,
                                            interpolation=cv2.INTER_NEAREST)

    def detect(self):
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        shrinkage_factor, transformed = self._mask()

        transformed = transformed
        labels, coords = model.predict_instances(normalize(transformed))

        _, height, width = self.data.shape

        nuclei_detected = cv2.resize(labels, dsize=(height, width),
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


def extract_centrioles(data, channel):
    """
    Extract the centrioles from the channel image.
    :param channel:
    :param data:
    :return: List of Points
    """
    focus_detector = CentriolesDetector(data, channel, 'Centriole')
    return focus_detector.detect()


def extract_nuclei(data, channel):
    """
    Extract the nuclei from the nuclei image.
    :param channel:
    :param data:
    :return: List of Contours.
    """
    nuclei_detector = NucleiDetector(data, channel, 'Nucleus')
    return nuclei_detector.detect()


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""
    result = cv2.pointPolygonTest(nucleus.contour,
                                  focus.centre,
                                  measureDist=True)
    return result


def assign(foci: list, nuclei: list, vicinity: int) -> list[
    tuple[Any, list[Any]]]:
    """
    Assign detected centrioles to the nearest nucleus.
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
            if distance > -50:
                assigned.append(f)
        pairs.append((n, assigned))

    return pairs


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='CENTRACK: Automatic centriole scoring')

    parser.add_argument('dataset',
                        type=Path,
                        help='path to the dataset')

    parser.add_argument('channel_nuclei',
                        type=int,
                        default=0,
                        help='channel id for nuclei segmentation, e.g., 0 or 4, default 0')

    return parser.parse_args()


def cli():
    args = parse_args()
    path_dataset = args.dataset
    dataset = DataSet(path_dataset)

    path_predictions = path_dataset / 'predictions'
    path_visualisation = path_dataset / 'visualisations'
    path_statistics = path_dataset / 'statistics'

    path_predictions.mkdir(exist_ok=True)
    path_visualisation.mkdir(exist_ok=True)
    path_statistics.mkdir(exist_ok=True)

    nuclei_channel = args.channel_nuclei
    if not dataset.projections.exists():
        raise FileExistsError(
            'Projection folder does not exist. Have you run `squash`?')
    fields = tuple(f for f in dataset.projections.glob('*.tif') if
                   not f.name.startswith('.'))

    scored = []

    for path in fields:
        logger_score.info('Loading %s', path.name)

        data = load_projection(path)
        channels, height, width = data.shape
        channels = list(range(channels))
        channels.pop(nuclei_channel)

        # This skips the print calls in spotipy
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            nuclei = extract_nuclei(data, nuclei_channel)
        mask_nuclei = np.zeros((height, width), dtype=np.uint8)
        for n in nuclei:
            color = n.idx
            n.draw(mask_nuclei, color=color, thickness=-1,
                   annotation=False)
            cv2.imwrite(str(path_predictions / f"{path.stem}_nuclei_preds.png"),
                        mask_nuclei)
        nuclei_plane = data[nuclei_channel, :, :]

        for i in channels:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                foci = extract_centrioles(data, i)
            if foci:
                foci_df = foci_prediction_prepare(foci, i)
                foci_df.to_csv(
                    path_predictions / f"{path.stem}_foci_{i}_preds.csv")
                logger_score.info('Detection in channel %s: %s nuclei, %s foci',
                                  i, len(nuclei), len(foci))
            else:
                logger_score.warning(
                    'No object were detected in channel %s: skipping...', i)
            assigned = assign(foci=foci,
                              nuclei=nuclei,
                              vicinity=-50)

            foci_plane = data[i, :, :]

            logger_score.debug('Creating annotation image...')
            background = prepare_background(nuclei_plane, foci_plane)
            annotation = draw_annotation(background, assigned, foci, nuclei)

            file_name = path.name.removesuffix(".tif")
            destination_path = path_visualisation / f'{file_name}_{i}_annot.png'
            successful = cv2.imwrite(str(destination_path), annotation)

            if successful:
                logger_score.debug('Saved at %s', destination_path)

            for pair in assigned:
                n, foci = pair
                scored.append({'fov': path.name,
                               'channel': i,
                               'nucleus': n.centre.position,
                               'score': len(foci),
                               })

    scores = pd.DataFrame(scored)
    binned = score_summary(scores)
    dst_statistics = str(path_statistics / f'statistics.csv')
    binned.to_csv(dst_statistics)
    logger_score.info('Analysis done.')


if __name__ == '__main__':
    cli()
