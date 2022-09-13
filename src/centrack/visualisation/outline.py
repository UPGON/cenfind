from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.draw import disk
from skimage.exposure import rescale_intensity


@dataclass(eq=True, frozen=True)
class ROI(ABC):
    """Abstract class to represent any region of interest"""

    @property
    @abstractmethod
    def centre(self):
        pass

    @abstractmethod
    def draw(self, plane, color, marker_type, marker_size):
        pass


@dataclass(eq=True, frozen=True)
class Centre(ROI):
    position: tuple
    idx: int = 0
    label: str = ''
    confidence: float = 0

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
                        org=(r, c + offset_col),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.4, thickness=1, color=color)

        return cv2.drawMarker(image, (r, c), color, markerType=marker_type,
                              markerSize=marker_size)

    def to_numpy(self):
        return np.asarray(self.centre)


@dataclass(eq=True, frozen=True)
class Contour(ROI):
    """Represent a blob using the row-column scheme."""

    contour: np.ndarray
    label: str = ''
    idx: int = 0
    confidence: float = 0

    @property
    def centre(self):
        moments = cv2.moments(self.contour)

        centre_x = int(moments['m01'] / (moments['m00'] + 1e-5))
        centre_y = int(moments['m10'] / (moments['m00'] + 1e-5))
        centre_r, centre_c = centre_y, centre_x
        return Centre((centre_r, centre_c), self.idx, self.label,
                      self.confidence)

    def draw(self, image, color=(0, 255, 0), annotation=True, thickness=2, **kwargs):
        r, c = self.centre.centre
        cv2.drawContours(image, [self.contour], -1, color, thickness=thickness)
        if annotation:
            cv2.putText(image, f'{self.label}{self.idx}',
                        org=(r, c),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.8, thickness=2, color=color)
            cv2.drawMarker(image, (r, c), (0, 0, 255),
                           markerType=cv2.MARKER_STAR,
                           markerSize=10)
        return image


def draw_foci(data: np.ndarray, foci: np.ndarray) -> np.ndarray:
    mask = np.zeros(data.shape, dtype='uint8')
    for r, c in foci:
        rr, cc = disk((r, c), 4)
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


def create_vignette(projection, marker_index: int, nuclei_index: int):
    """
    Normalise all markers
    Represent them as blue
    Highlight the channel in green
    :param projection:
    :param nuclei_index:
    :param marker_index:
    :return:
    """
    layer_nuclei = projection.projection[nuclei_index, :, :]
    layer_marker = projection.projection[marker_index, :, :]

    nuclei = _color_channel(layer_nuclei, (1, 0, 0), 'uint8')
    marker = _color_channel(layer_marker, (0, 1, 0), 'uint8')

    res = cv2.addWeighted(marker, 1, nuclei, .2, 50)
    res = cv2.putText(res, f"{projection.name} {marker_index}",
                      (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      .8, (255, 255, 255), 1, cv2.LINE_AA)

    return res
