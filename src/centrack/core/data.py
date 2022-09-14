import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tifffile as tf


def extract_info(pattern: re, dataset_name: str):
    res = re.match(pattern, dataset_name).groupdict()
    markers = res['markers'].split('+')
    res['markers'] = tuple(markers)

    return res


@dataclass
class Dataset:
    """
    Represents a dataset structure
    """
    path: Union[str, Path]
    image_type: str = '.ome.tif'

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.projections = self.path / 'projections'
        self.projections.mkdir(exist_ok=True)

        self.predictions = self.path / 'predictions'
        self.predictions.mkdir(exist_ok=True)

        self.visualisation = self.path / 'visualisations'
        self.visualisation.mkdir(exist_ok=True)

        self.statistics = self.path / 'statistics'
        self.statistics.mkdir(exist_ok=True)

    def _read_split(self, split_type) -> List[Tuple[str, int]]:
        with open(self.path / f'{split_type}.txt', 'r') as f:
            files = f.read().splitlines()
        files = [f.split(',') for f in files if f]
        files = [(str(f[0]), int(f[1])) for f in files]

        return files

    def fields(self, split: str = None) -> List[Tuple[str, int]]:
        """
        Fetch the fields of view for train or test
        :param split: all, test or train
        :return: a list of tuples (fov name, channel id)
        """

        if split is None:
            return self._read_split('train') + self._read_split('test')
        else:
            return self._read_split(split)


@dataclass
class Field:
    name: str
    dataset: Dataset

    @property
    def stack(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'raw' / f"{self.name}.ome.tif"))

    @property
    def projection(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'projections' / f"{self.name}_max.tif"))

    def channel(self, channel: int) -> np.ndarray:
        return self.projection[channel, :, :]

    def annotation(self, channel) -> np.ndarray:
        name = f"{self.name}_max_C{channel}"
        path_annotation = self.dataset.path / 'annotations' / 'centrioles' / f"{name}.txt"
        if path_annotation.exists():
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=',')
            return annotation
        else:
            raise FileNotFoundError(f"{path_annotation}")

    def mask(self, channel) -> np.ndarray:
        mask_name = f"{self.name}_max_C{channel}.tif"
        path_annotation = self.dataset.path / 'annotations' / 'cells' / mask_name
        if path_annotation.exists():
            return tf.imread(str(path_annotation))
        else:
            raise FileNotFoundError(path_annotation)
