import logging
from pathlib import Path
from dataclasses import dataclass

from cv2 import cv2
import numpy as np
import xarray as xr

import tifffile as tf

from scipy.ndimage import maximum_filter, minimum_filter


def contrast(data):
    return cv2.convertScaleAbs(data, alpha=255 / data.max())


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

    @property
    def data(self):
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

    def select(self, marker):
        data = self.data.loc[marker].to_numpy()
        return Channel(self, data)


@dataclass
class Channel:
    field: Field
    data: np.ndarray

    def maximum_filter(self, size):
        return Channel(self.field, maximum_filter(self.data, size=(size, size)))

    def threshold(self, threshold=0):
        if threshold:
            _, mask = cv2.threshold(self.data, threshold, 255, cv2.THRESH_BINARY)
            return Channel(self.field, mask)
        else:
            _, mask = cv2.threshold(self.data, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Channel(self.field, mask)
