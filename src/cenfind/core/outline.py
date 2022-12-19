from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.draw import disk
from skimage.exposure import rescale_intensity

from cenfind.core.data import Field


@dataclass(eq=True, frozen=False)
class ROI(ABC):
    """Abstract class to represent any region of interest"""

    @property
    @abstractmethod
    def centre(self):
        pass

    @abstractmethod
    def draw(self, plane, color, marker_type, marker_size):
        pass


@dataclass(eq=True, frozen=False)
class Centre(ROI):
    position: tuple
    idx: int = 0
    label: str = ''
    confidence: float = 0
    parent: 'Centre' = None

    @property
    def centre(self):
        row, col = self.position
        return int(row), int(col)

    def draw(self, image, color=(0, 255, 0), annotation=True,
             marker_type=cv2.MARKER_SQUARE, marker_size=8):
        r, c = self.centre
        offset_col = int(.01 * image.shape[1])

        if annotation:
            cv2.putText(image, f'{self.label} {self.idx}',
                        org=(c + offset_col, r),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.4, thickness=1, color=color)

        return cv2.drawMarker(image, (c, r), color, markerType=marker_type,
                              markerSize=marker_size)

    def to_numpy(self):
        return np.asarray(self.centre)

    def to_cv2(self):
        y, x = self.centre
        return x, y


@dataclass(eq=True, frozen=False)
class Contour(ROI):
    """Represent a blob using the row-column scheme."""

    contour: np.ndarray
    label: str = ''
    idx: int = 0
    confidence: float = 0

    @property
    def centre(self):
        moments = cv2.moments(self.contour)

        centre_x = int(moments['m10'] / (moments['m00'] + 1e-5))
        centre_y = int(moments['m01'] / (moments['m00'] + 1e-5))
        return Centre((centre_y, centre_x), self.idx, self.label,
                      self.confidence)

    def draw(self, image, color=(0, 255, 0), annotation=True, thickness=2, **kwargs):
        r, c = self.centre.centre
        cv2.drawContours(image, [self.contour], -1, color, thickness=thickness)
        if annotation:
            cv2.putText(image, f'{self.label}{self.idx}',
                        org=(c, r),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.8, thickness=2, color=color)
            cv2.drawMarker(image, (c, r), color,
                           markerType=cv2.MARKER_STAR,
                           markerSize=10)
        return image


def resize_image(data, factor=256):
    height, width = data.shape
    shrinkage_factor = int(height // factor)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(data,
                              dsize=(height_scaled, width_scaled),
                              fx=1, fy=1,
                              interpolation=cv2.INTER_NEAREST)
    return data_resized

def draw_foci(data: np.ndarray, foci: list[Centre], radius=.4, pixel_size=.1025) -> np.ndarray:
    mask = np.zeros(data.shape, dtype='uint8')
    for f in foci:
        r, c = f.to_numpy()
        rr, cc = disk((r, c), int(radius/pixel_size))
        try:
            mask[rr, cc] = 250
        except IndexError:
            continue
    return mask


def _color_channel(data, color, out_range):
    """
    Create a colored version of a channel image
    :return:
    """
    data = rescale_intensity(data, out_range=out_range)
    b = np.multiply(data, color[0], casting='unsafe')
    g = np.multiply(data, color[1], casting='unsafe')
    r = np.multiply(data, color[2], casting='unsafe')
    res = cv2.merge([b, g, r])
    return res


def create_vignette(field: Field, marker_index: int, nuclei_index: int):
    """
    Normalise all markers
    Represent them as blue
    Highlight the channel in green
    :param field:
    :param nuclei_index:
    :param marker_index:
    :return:
    """
    layer_nuclei = field.channel(nuclei_index)
    layer_marker = field.channel(marker_index)

    nuclei = _color_channel(layer_nuclei, (1, 0, 0), 'uint8')
    marker = _color_channel(layer_marker, (0, 1, 0), 'uint8')

    res = cv2.addWeighted(marker, 1, nuclei, .5, 0)
    res = cv2.putText(res, f"{field.name} {marker_index}",
                      (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                      .8, (255, 255, 255), 1, cv2.LINE_AA)

    return res
