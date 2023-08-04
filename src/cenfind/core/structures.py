from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from cenfind.core.log import get_logger

logger = get_logger(__name__)


@dataclass(eq=True, frozen=False)
class ROI(ABC):
    """Abstract class to represent any region of interest"""

    @property
    @abstractmethod
    def centre(self):
        pass

    @abstractmethod
    def intensity(self, image):
        pass


@dataclass(eq=True, frozen=False)
class Point(ROI):
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

    def intensity(self, image: np.ndarray, channel: int = None):
        if image.ndim == 3:
            return image[channel, self.centre]
        return image[self.centre]


@dataclass(eq=True, frozen=False)
class Contour(ROI):
    """Represent a blob using the row-column scheme."""

    contour: np.ndarray
    channel: int
    label: str = ""
    index: int = 0
    centrioles: list = field(default_factory=list)

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
