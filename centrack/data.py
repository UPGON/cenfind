import logging
from pathlib import Path
from dataclasses import dataclass

from cv2 import cv2
import numpy as np
import xarray as xr

import tifffile as tf

from scipy.ndimage import maximum_filter, minimum_filter


@dataclass
class Marker:
    position: int
    name: str


@dataclass
class PixelSize:
    value: float
    units: str

    def in_cm(self):
        conversion_map = {
            'um': 10e4,
            'μm': 10e4,
            'nm': 10e7,
        }
        return self.value / conversion_map[self.units]


@dataclass
class Condition:
    markers: list
    genotype: str
    pixel_size: PixelSize


@dataclass
class DataSet:
    path: Path
    condition: Condition

    @property
    def projections(self):
        """Define the path to projections folder."""
        return self.path / 'projections'

    @property
    def raw(self):
        """Define the path to raw folder."""
        return self.path / 'raw'


@dataclass
class Field:
    path: Path
    dataset: DataSet

    @property
    def markers(self):
        return self.dataset.condition.markers

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        logging.info('Loading %s', self.path)

        with tf.TiffFile(self.path) as file:
            data = file.asarray()
            data = np.squeeze(data)

        result = xr.DataArray(data,
                              dims=['channel', 'width', 'height'],
                              coords={'channel': self.markers})

        return result


@dataclass
class Channel:
    data: xr.DataArray

    def __getitem__(self, item):
        return self.data.loc[item, :, :]
