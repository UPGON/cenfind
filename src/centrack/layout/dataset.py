import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Tuple

import cv2
import numpy as np
import tifffile as tf
from skimage.exposure import rescale_intensity
from spotipy.utils import normalize_fast2d


@dataclass
class DataSet:
    """
    Represents a dataset structure
    """
    path: Path

    @property
    def name(self):
        return self.path.name

    @property
    def raw(self):
        """Define the path to raw folder."""
        return self.path / 'raw'

    @property
    def projections(self):
        """Define the path to projections folder."""
        return self.path / 'projections'

    @property
    def vignettes(self):
        """Define the path to the vignettes' folder."""
        return self.path / 'vignettes'

    @property
    def annotations(self):
        return self.path / 'annotations'

    @property
    def predictions(self):
        return self.path / 'predictions'

    @property
    def visualisation(self):
        return self.path / 'visualisation'

    def splits(self, suffix, p=.9) -> Tuple[List, List]:
        """
        Assign the FOV between train and test
        :param p: the fraction of train examples, by default .9
        :param suffix: the type of raw files, by default .ome.tif
        :return: a tuple of lists
        """
        random.seed(1993)

        files = fetch_files(self.raw, suffix)
        file_stems = [f.name.removesuffix(suffix) for f in files]
        size = len(file_stems)
        split_idx = int(p * size)
        shuffled = random.sample(file_stems, k=size)
        split_test = shuffled[split_idx:]
        split_train = shuffled[:split_idx]
        return split_train, split_test

    def split_images_channel(self, split_type):
        with open(self.path / f'{split_type}_channels.txt', 'r') as f:
            files = f.read().splitlines()
        files = [f.split(',') for f in files if f]
        return files


@dataclass
class FieldOfView:
    """
    Representation of a projection (CxHxW)
    """
    dataset: DataSet
    name: str

    @property
    def data(self) -> np.array:
        return tf.imread(str(self.dataset.path / 'projections' / f"{self.name}_max.tif"))

    def load_channel(self, channel: int):
        return self.data[channel, :, :]

    def load_annotation(self, channel):
        path_annotation = self.dataset.annotations / 'centrioles' / f"{self.name}_max_C{channel}.txt"
        if path_annotation.exists():
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=',')
            return annotation
        else:
            raise FileExistsError(f"{path_annotation}")

    def generate_vignette(self, marker_index: int, nuclei_index: int):
        """
        Normalise all markers
        Represent them as blue
        Highlight the channel in green
        :param nuclei_index:
        :param marker_index:
        :return:
        """
        layer_nuclei = self.load_channel(nuclei_index)
        layer_marker = self.load_channel(marker_index)

        nuclei = contrast_color(layer_nuclei, (1, 1, 1), 'uint8')
        marker = contrast_color(layer_marker, (0, 1, 0), 'uint8')

        res = cv2.addWeighted(marker, 1, nuclei, .2, 50)
        res = cv2.putText(res, f"{self.name} channel: {marker_index}",
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                          .8, (255, 255, 255), 2, cv2.LINE_AA)

        return res

    def detect_centrioles(self, channel_id, model, prob_threshold=.5, min_distance=2):
        """
        Detect centrioles
        :param channel_id:
        :param model:
        :param prob_threshold:
        :param min_distance:
        :return:
        """
        foci = self.load_channel(channel_id)
        foci = normalize_fast2d(foci)
        probs, points = model.predict(foci, prob_thresh=prob_threshold, min_distance=min_distance)

        return probs, points

    def _combine_other_layers(self, nuclei_index, marker_index):
        channels_n = self.data.shape[0]
        used = {marker_index, nuclei_index}
        other_channels = set(range(channels_n)).difference(used)
        layers = [rescale_intensity(self.load_channel(l), out_range='uint8')
                  for l in other_channels]
        other_layers = np.stack(layers, axis=0).max(axis=0)
        return other_layers


def fetch_files(path_source: Path, file_type):
    """
    Create a list of files
    :param path_source:
    :param file_type:
    :return:
    """
    if not path_source.exists():
        raise FileExistsError(path_source)
    pattern = f'*{file_type}'
    files_generator = path_source.rglob(pattern)

    return [file for file in files_generator if not file.name.startswith('.')]


def color_channel(layer, color):
    """
    Create a colored version of a channel image
    :param layer:
    :param color:
    :return:
    """
    b = np.multiply(layer, color[0], casting='unsafe')
    g = np.multiply(layer, color[1], casting='unsafe')
    r = np.multiply(layer, color[2], casting='unsafe')
    res = cv2.merge([b, g, r])
    return res


def contrast_color(data, color, out_range):
    res = rescale_intensity(data, out_range=out_range)
    # res = cv2.equalizeHist(res)
    res = color_channel(res, color)
    return res
