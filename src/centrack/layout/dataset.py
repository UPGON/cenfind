import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tifffile as tf

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


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

    def splits(self, p=.9, suffix='.ome.tif') -> Tuple[List, List]:
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


def build_name(path: Path, projection_type='max') -> str:
    """
    Extract the file name, remove the suffixes and append the projection type.
    :param path:
    :param channel:
    :param projection_type: the type of projection, by default max
    :return: file name of the projection
    """
    file_name = path.name
    suffixes = ''.join(path.suffixes)
    file_name_no_suffix = file_name.removesuffix(suffixes)
    return file_name_no_suffix + f'_{projection_type}.tif'


def fetch_files(path_source: Path, file_type):
    """
    Create a list of files.
    :param path_source:
    :param file_type:
    :return:
    """
    if not path_source.exists():
        raise FileExistsError(path_source)
    pattern = f'*{file_type}'
    files_generator = path_source.rglob(pattern)

    return [file for file in files_generator if not file.name.startswith('.')]


@dataclass
class FieldOfView:
    """
    Representation of a projection (CxHxW)
    """
    dataset: DataSet
    name: str

    @property
    def name(self):
        return self.name

    @property
    def data(self) -> np.array:
        return tf.imread(str(self.dataset.path / 'projections' / f"{self.name}_max.tif"))

    def load_channel(self, channel_id):
        return self.data[channel_id, :, :]

    def load_annotation(self, channel_id):
        path_annotation = self.dataset.annotations / 'centrioles' / f"{self.name}_C{channel_id}.txt"

        try:
            annotation = np.loadtxt(path_annotation, dtype=int, delimiter=',')
        except FileNotFoundError:
            raise f'Annotation not found for {path_annotation}'

        return annotation
