import logging
import re
from pathlib import Path

import numpy as np
import tifffile as tf
import xarray as xr
from dataclasses import dataclass


@dataclass
class Marker:
    """Represents a marker."""
    protein: str = None
    channel: str = None
    position: int = None
    wave_length: int = None
    code: str = None

    @property
    def _code(self):
        if self.code is not None:
            return self.code
        else:
            return f'{self.channel}{self.protein}{self.wave_length}'

    @classmethod
    def from_str(cls, code, pattern=r'([rgbm])([\w\d]+)', position=None):
        if code is None:
            raise ValueError('Provide a code')
        if code == 'DAPI':
            return cls(protein='DNA',
                       channel='b',
                       position=0
                       )
        else:
            remainder, wave_length = code[:-3], code[-3:]
            res = re.match(pattern, remainder)
            if res is None:
                raise ValueError(f'Regex unsuccessful: {res=}')
            channel, protein = res.groups()
            return cls(protein=protein,
                       channel=channel,
                       position=position,
                       wave_length=wave_length)


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
