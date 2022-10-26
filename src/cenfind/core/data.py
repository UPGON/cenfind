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
    projection_suffix: str = '_max'

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
        self._write_fields()

    def _write_fields(self):
        """
        Write field names to fields.txt.
        """
        if (self.path / 'raw').exists():
            folder = self.path / 'raw'
        elif (self.path / 'projections').exists():
            folder = self.path / 'projections'
        else:
            raise FileNotFoundError(self.path)

        fields = []
        for f in folder.iterdir():
            if f.name.startswith('.'):
                continue
            fields.append(f.name.split('.')[0].rstrip(self.projection_suffix))

        with open(self.path / 'fields.txt', 'w') as f:
            for field in fields:
                f.write(field + '\n')

    @property
    def fields(self):
        fields_path = self.path / 'fields.txt'
        with open(fields_path, 'r') as f:
            return f.read().splitlines()

    def _read_split(self, split_type, channel_id=None) -> List[Tuple[str, int]]:
        with open(self.path / f'{split_type}.txt', 'r') as f:
            files = f.read().splitlines()
        files = [f.split(',') for f in files if f]
        if channel_id:
            return [(str(f[0]), channel_id) for f in files]
        else:
            return [(str(f[0]), int(f[1])) for f in files]

    def pairs(self, split: str = None, channel_id: int = None) -> List[Tuple[str, int]]:
        """
        Fetch the fields of view for train or test
        :param channel_id:
        :param split: all, test or train
        :return: a list of tuples (fov name, channel id)
        """

        if split is None:
            return self._read_split('train', channel_id) + self._read_split('test', channel_id)
        else:
            return self._read_split(split, channel_id)


@dataclass
class Field:
    name: str
    dataset: Dataset

    @property
    def stack(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'raw' / f"{self.name}.ome.tif"))

    @property
    def projection(self) -> np.ndarray:
        return tf.imread(str(self.dataset.path / 'projections' / f"{self.name}{self.dataset.projection_suffix}.tif"))

    def channel(self, channel: int) -> np.ndarray:
        return self.projection[channel, :, :]

    def annotation(self, channel) -> np.ndarray:
        """
        Load annotation file from text file given channel
        loaded as row col, row major.
        ! the text format is x, y; origin at top left;
        :param channel:
        :return:
        """
        name = f"{self.name}{self.dataset.projection_suffix}_C{channel}"
        path_annotation = self.dataset.path / 'annotations' / 'centrioles' / f"{name}.txt"
        if path_annotation.exists():
            annotation = np.loadtxt(str(path_annotation), dtype=int, delimiter=',')
            return annotation[:, [1, 0]]
        else:
            raise FileNotFoundError(f"{path_annotation}")

    def mask(self, channel) -> np.ndarray:
        mask_name = f"{self.name}{self.dataset.projection_suffix}_C{channel}.tif"
        path_annotation = self.dataset.path / 'annotations' / 'cells' / mask_name
        if path_annotation.exists():
            return tf.imread(str(path_annotation))
        else:
            raise FileNotFoundError(path_annotation)
