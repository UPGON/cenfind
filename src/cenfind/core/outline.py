from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from skimage.draw import disk
from skimage.exposure import rescale_intensity

from cenfind.core.data import Field
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
    def draw(self, plane, color, marker_type, marker_size):
        pass


@dataclass(eq=True, frozen=False)
class Centre(ROI):
    position: tuple
    idx: int = 0
    label: str = ""
    confidence: float = 0
    parent: "Centre" = None

    @property
    def centre(self):
        row, col = self.position
        return int(row), int(col)

    def draw(
            self,
            image,
            color=(0, 255, 0),
            annotation=True,
            marker_type=cv2.MARKER_SQUARE,
            marker_size=8,
    ):
        r, c = self.centre
        offset_col = int(0.01 * image.shape[1])

        if annotation:
            cv2.putText(
                image,
                f"{self.label} {self.idx}",
                org=(c + offset_col, r),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=color,
            )

        return cv2.drawMarker(
            image, (c, r), color, markerType=marker_type, markerSize=marker_size
        )

    def to_numpy(self):
        return np.asarray(self.centre)

    def to_cv2(self):
        y, x = self.centre
        return x, y


@dataclass(eq=True, frozen=False)
class Contour(ROI):
    """Represent a blob using the row-column scheme."""

    contour: np.ndarray
    label: str = ""
    idx: int = 0
    confidence: float = 0
    centrioles: list = field(default_factory=list)

    @property
    def centre(self):
        moments = cv2.moments(self.contour)

        centre_x = int(moments["m10"] / (moments["m00"] + 1e-5))
        centre_y = int(moments["m01"] / (moments["m00"] + 1e-5))
        return Centre((centre_y, centre_x), self.idx, self.label, self.confidence)

    def draw(self, image, color=(0, 255, 0), annotation=True, thickness=2, **kwargs):
        r, c = self.centre.centre
        cv2.drawContours(image, [self.contour], -1, color, thickness=thickness)
        if annotation:
            cv2.putText(
                image,
                f"{self.label}{self.idx}",
                org=(c, r),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                thickness=2,
                color=color,
            )
            cv2.drawMarker(
                image, (c, r), color, markerType=cv2.MARKER_STAR, markerSize=10
            )
        return image

    def add_centrioles(self, centriole: Centre):
        self.centrioles.append(centriole)
        return 0


def resize_image(data: np.ndarray, factor: int = 256) -> np.ndarray:
    """
    Resize the image for nuclei segmentation by StarDist
    :param data: a single channel image (H x W)
    :param factor: the target dimension
    :return the image resized
    """
    height, width = data.shape
    shrinkage_factor = int(height // factor)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(
        data,
        dsize=(height_scaled, width_scaled),
        fx=1,
        fy=1,
        interpolation=cv2.INTER_NEAREST,
    )
    return data_resized


def draw_foci(data: np.ndarray, foci: list[Centre], radius=2) -> np.ndarray:
    """
    Draw foci as disks of given radius
    :param data: the channel for the dimension extraction
    :param foci: the list of foci
    :param radius: the radius of foci
    :return the mask with foci
    """
    mask = np.zeros(data.shape, dtype="uint8")
    for f in foci:
        rr, cc = disk(f.to_numpy(), radius, shape=data.shape[-2:])
        mask[rr, cc] = 250
    return mask


def _color_channel(data: np.ndarray, color: tuple, out_range: str):
    """
    Create a colored version of a channel image
    :param data: the data to use
    :param color: the color as a tuple (B, G, R)
    :param out_range:
    :return:
    """
    data = rescale_intensity(data, out_range=out_range)
    b = np.multiply(data, color[0], casting="unsafe")
    g = np.multiply(data, color[1], casting="unsafe")
    r = np.multiply(data, color[2], casting="unsafe")
    res = cv2.merge([b, g, r])
    return res


def create_vignette(field: Field, marker_index: int, nuclei_index: int):
    """
    Normalise all markers
    sRepresent them as blue
    Highlight the channel in green
    :param field:
    :param nuclei_index:
    :param marker_index:
    :return:
    """
    layer_nuclei = field.channel(nuclei_index)
    layer_marker = field.channel(marker_index)

    nuclei = _color_channel(layer_nuclei, (1, 0, 0), "uint8")
    marker = _color_channel(layer_marker, (0, 1, 0), "uint8")

    res = cv2.addWeighted(marker, 1, nuclei, 0.5, 0)
    res = cv2.putText(
        res,
        f"{field.name} {marker_index}",
        (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return res


def visualisation(
        field: Field,
        nuclei: list,
        centrioles: list,
        channel_centrioles: int,
        channel_nuclei: int,
) -> np.ndarray:
    background = create_vignette(
        field, marker_index=channel_centrioles, nuclei_index=channel_nuclei
    )

    if nuclei is None:
        return background

    for nucleus in nuclei:
        background = nucleus.draw(background, annotation=False)
        background = nucleus.centre.draw(background, annotation=False)
        for centriole in centrioles:
            background = centriole.draw(background, annotation=False)

        for centriole in nucleus.centrioles:
            cv2.arrowedLine(
                background,
                centriole.to_cv2(),
                nucleus.centre.to_cv2(),
                color=(0, 255, 0),
                thickness=1,
            )

    return background
