from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tf
import xarray as xr

from centrack.describe import Condition


@dataclass
class Field:
    path: Path
    condition: Condition

    @property
    def markers(self):
        return self.condition.markers

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)

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


def is_tif(filename):
    _filename = str(filename)
    return _filename.endswith('.tif') and not _filename.startswith('.')


def build_name(path):
    """
    Remove the suffixes and append `_max`.
    :param path:
    :return:
    """
    file_name = path.name
    suffixes = ''.join(path.suffixes)
    file_name_no_suffix = file_name.removesuffix(suffixes)
    return file_name_no_suffix + '_max' + '.tif'



def write_projection(dst, data, pixel_size=None):
    """
    Writes the projection to the disk.
    """
    if pixel_size:
        res = (1 / pixel_size, 1 / pixel_size, 'CENTIMETER')
    else:
        res = None
    tf.imwrite(dst, data, photometric='minisblack', resolution=res)
