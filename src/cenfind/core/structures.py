from abc import ABC, abstractmethod
from attrs import define, field, astuple

import cv2
import numpy as np

from cenfind.core.log import get_logger

logger = get_logger(__name__)


@define
class Point:
    position: tuple
    channel: int
    index: int = 0
    label: str = ""
    parent: "Point" = None

    @property
    def centre(self):
        row, col = self.position
        return int(row), int(col)

    def to_cv2(self):
        y, x = self.centre
        return x, y

    def intensity(self, image: np.ndarray, k: int = 0, channel: int = None):
        r, c = self.centre
        max_r, max_c = image.shape[-2:]

        r_start = np.clip(r - k, 0, max_r)
        r_stop = np.clip(r + k, 0, max_r)
        c_start = np.clip(c - k, 0, max_c)
        c_stop = np.clip(c + k, 0, max_c)

        if image.ndim < 3:
            return np.sum(image[r_start:r_stop, c_start:c_stop])
        if channel is None:
            raise ValueError('Channel must be supplied when image has 3 dimensions')
        return np.sum(image[channel, r_start:r_stop, c_start:c_stop])


@define
class Contour:
    """Represent a blob using the row-column scheme."""
    contour: np.ndarray
    channel: int
    label: str = ""
    index: int = 0
    centrioles: list = []

    @property
    def centre(self):
        moments = cv2.moments(self.contour)

        centre_x = int(moments["m10"] / (moments["m00"] + 1e-5))
        centre_y = int(moments["m01"] / (moments["m00"] + 1e-5))
        return Point((centre_y, centre_x), self.channel, self.index, self.label)

    def intensity(self, image: np.ndarray):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [self.contour], 0, 255, -1)
        masked = cv2.bitwise_and(image, mask)
        return np.sum(masked)

    def area(self):
        return cv2.contourArea(self.contour)
