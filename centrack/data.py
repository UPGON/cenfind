from pathlib import Path

import cv2
import numpy as np

from aicsimageio import AICSImage

from scipy.ndimage import maximum_filter, minimum_filter


def contrast(data):
    return cv2.convertScaleAbs(data, alpha=255 / data.max())


class FileName:
    """Model a file name"""

    def __init__(self, genotype, markers, position, extension):
        self.genotype = genotype
        self.markers = markers
        self._position = position
        self.extension = extension

    @property
    def position(self):
        r, c = self.position
        return f"{r:03}_{c:03}"

    @property
    def condition(self):
        return '_'.join([self.genotype, *self.markers, ])

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, extension):
        self._extension = extension

    @property
    def filename(self):
        return self.condition + '_' + self.position + '.' + self.extension

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.filename})"


class DataSet:
    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self):
        return str(self.path.name)

    @property
    def raw(self):
        """Define the path to raw folder."""
        return Path(self.path) / 'raw'

    @property
    def projections(self):
        """Define the path to projections folder."""
        return Path(self.path) / 'projections'

    @property
    def fields(self):
        return [p for p in (self.path / 'raw').glob('*.ome.tif') if not p.name.startswith('.')]

    def markers(self, marker_sep='+'):
        """
        Extract the markers' name from the dataset string.
        The string must follows the structure `<genotype>_marker1+marker2`
        It append the DAPI at the beginning of the list.

        :param marker_sep:
        :param self:
        :return: a dictionary of markers
        """

        markers = self.path.name.split('_')[-2].split(marker_sep)
        if 'DAPI' not in markers:
            markers = ['DAPI'] + markers

        return {k: v for k, v in enumerate(markers)}


class Field:
    def __init__(self, data, dataset):
        self.data = data
        self.dataset = dataset

    def select_plane(self, channel_id):
        return Plane(self.data.get_image_data("YX", C=channel_id), self)


class Plane:
    def __init__(self, data, field):
        self.data = data
        self.field = field

    def __repr__(self):
        return "Plane"

    @property
    def dims(self):
        return self.data.shape

    @property
    def height(self):
        return self.dims[0]

    @property
    def width(self):
        return self.dims[1]

    @property
    def is_mask(self):
        unique_values = np.unique(self.data)
        return len(unique_values) < 2

    def write_png(self, path):
        cv2.imwrite(str(path), self.data)

    def contrast(self):
        return Plane(contrast(self.data), self.field)

    def blur_median(self, ks):
        return Plane(cv2.medianBlur(self.data, ks), self.field)

    def maximum_filter(self, size):
        return Plane(maximum_filter(self.data, size=(size, size)), self.field)

    def minimum_filter(self, size):
        return Plane(minimum_filter(self.data, size=(size, size)), self.field)

    def threshold(self, threshold=0):
        if threshold:
            _, mask = cv2.threshold(self.data, threshold, 255, cv2.THRESH_BINARY)
            return Plane(mask, self.field)
        else:
            _, mask = cv2.threshold(self.data, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Plane(mask, self.field)
