from abc import ABC, abstractmethod

import numpy as np
from cv2 import cv2
from dataclasses import dataclass


def color_scale(color):
    return tuple(c / 255. for c in color)


@dataclass(eq=True, frozen=True)
class ROI(ABC):
    """Abstract class to represent any region of interest"""

    @property
    @abstractmethod
    def dims(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def centre(self):
        pass

    @property
    @abstractmethod
    def bbox(self):
        pass

    @abstractmethod
    def draw(self, plane, color, marker_type, marker_size):
        pass

    @abstractmethod
    def extract(self, plane):
        pass


@dataclass(eq=True, frozen=True)
class Centre(ROI):
    position: tuple
    idx: int = 0
    label: str = ''
    confidence: float = 0

    @property
    def row(self):
        return self.position[0]

    @property
    def col(self):
        return self.position[1]

    @property
    def dims(self):
        return 0, 0

    @property
    def height(self):
        return 0

    @property
    def width(self):
        return 0

    @property
    def centre(self):
        return self.row, self.col

    @property
    def bbox(self):
        top_left = self.row - 32, self.col - 32
        bottom_right = self.row + 32, self.col + 32
        return BBox(top_left, bottom_right, self.idx, self.label, self.confidence)

    def draw(self, image, color=(0, 255, 0), annotation=True, marker_type=cv2.MARKER_SQUARE, marker_size=8):
        r, c = self.centre
        offset_col = int(.01 * image.shape[1])

        if annotation:
            cv2.putText(image, f'{self.label} {self.idx}',
                        org=(r, c + offset_col),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.4, thickness=1, color=color)

        return cv2.drawMarker(image, (r, c), color, markerType=marker_type,
                              markerSize=marker_size)

    def extract(self, plane):
        # return plane[self.row - 32:self.col - 32, self.row + 32:self.col + 32]
        return self.bbox.extract(plane)


@dataclass(eq=True, frozen=True)
class BBox(ROI):
    top_left: tuple
    bottom_right: tuple
    idx: int = 0
    label: str = ''
    confidence: float = 0

    @property
    def dims(self):
        return self.height, self.width

    @property
    def height(self):
        start_row, _ = self.top_left
        stop_row, _ = self.bottom_right
        return stop_row - start_row

    @property
    def width(self):
        _, start_col = self.top_left
        _, stop_col = self.bottom_right
        return stop_col - start_col

    @property
    def centre(self):
        """Compute the centre (r_centre, c_centre)"""
        r_centre = (self.bottom_right[0] + self.top_left[0]) // 2
        c_centre = (self.bottom_right[1] + self.top_left[1]) // 2
        return Centre((r_centre, c_centre), self.idx, self.label, self.confidence)

    @property
    def bbox(self):
        return self

    def draw(self, image, color=(0, 255, 0), annotation=True, **kwargs):
        """Draw bounding box on an image."""
        offset = int(.01 * image.width)
        r, c = self.centre.centre
        cv2.rectangle(image, self.top_left, self.bottom_right,
                      color, thickness=kwargs['thickness'])
        if annotation:
            cv2.putText(image, f'{self.label} {self.idx}',
                        org=(r + offset, c),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, thickness=1, color=color)

        return image

    def extract(self, image):
        start_row, start_col = self.top_left
        stop_row, stop_col = self.bottom_right
        return image[start_row:stop_row, start_col:stop_col]


@dataclass(eq=True, frozen=True)
class Contour(ROI):
    """Represent a blob using the row-column scheme."""

    contour: np.ndarray
    label: str = ''
    idx: int = 0
    confidence: float = 0

    @property
    def dims(self):
        _, _, h, w = cv2.boundingRect(self.contour)  # (row, col, h, w)
        return h, w

    @property
    def height(self):
        return self.dims[0]

    @property
    def width(self):
        return self.dims[1]

    @property
    def bbox(self):
        r, c, h, w = cv2.boundingRect(self.contour)  # (r, c, h, w)
        top_left = r, c
        bottom_right = r + h, c + w

        return BBox(top_left, bottom_right, self.idx, self.label, self.confidence)

    @property
    def centre(self):
        moments = cv2.moments(self.contour)

        centre_x = int(moments['m01'] / (moments['m00'] + 1e-5))
        centre_y = int(moments['m10'] / (moments['m00'] + 1e-5))
        centre_r, centre_c = centre_y, centre_x
        return Centre((centre_r, centre_c), self.idx, self.label, self.confidence)

    def draw(self, image, color=(0, 255, 0), annotation=True, **kwargs):
        r, c = self.centre.centre
        cv2.drawContours(image, [self.contour], -1, color, thickness=2)
        if annotation:
            cv2.putText(image, f'{self.label}{self.idx}',
                        org=(r, c),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.8, thickness=2, color=color)
            cv2.drawMarker(image, (r, c), (0, 0, 255), markerType=cv2.MARKER_STAR,
                           markerSize=10)
        return image

    def extract(self, plane):
        return self.bbox.extract(plane)


if __name__ == '__main__':
    print(0)
