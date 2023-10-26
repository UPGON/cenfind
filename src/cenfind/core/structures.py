from typing import Tuple

import cv2
import numpy as np
from attrs import define

from cenfind.core.data import Field
from cenfind.core.log import get_logger

logger = get_logger(__name__)


@define
class Centriole:
    """
    Represents a centriole as a 2d point associated with a Field and a channel index.

    The parent attributes is used to ensure that neighbouring centrioles are assigned to the
    same nucleus.

    Attributes:
        field: Field used for centriole detection.
        channel: Channel used for centriole detection.
        centre: Coordinates (row major) of the detected centriole.
        index: Index of centriole.
        label: String label
        parent: Parent centriole used for tying neighbouring centrioles into a centrosome.
    """
    field: Field
    channel: int
    centre: Tuple[int, int]
    index: int = 0
    label: str = ""
    parent: "Centriole" = None

    @property
    def centre_xy(self) -> Tuple[int, int]:
        """
        Represent a row-major point as an XY point.
        Returns:

        """
        y, x = self.centre

        return x, y

    def intensity(self, image: np.ndarray, k: int = 0, channel: int = None) -> int:
        """
        Computes the intensity of the focus within the k-pixel neighbourhood on the image provided.
        Args:
            image: The image for signal extraction.
            k: Pixel (radius) neighbourhood for signal extraction
            channel: Channel to use for signal extraction.

        Returns:

        """
        r, c = self.centre
        max_r, max_c = image.shape[-2:]

        r_start = np.clip(r - k, 0, max_r)
        r_stop = np.clip(r + k, 0, max_r)
        c_start = np.clip(c - k, 0, max_c)
        c_stop = np.clip(c + k, 0, max_c)

        if image.ndim < 3:
            return int(np.sum(image[r_start:r_stop, c_start:c_stop]))

        if channel is None:
            raise ValueError("Channel must be supplied when image has 3 dimensions")

        return int(np.sum(image[channel, r_start:r_stop, c_start:c_stop]))

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "pos_r": self.centre[0],
            "pos_c": self.centre[1],
            "intensity": self.intensity(self.field.data[self.channel, ...], k=1)
        }


@define
class Nucleus:
    """
    Represents a Nucleus with a contour.

    Attributes:
        field: Field of view
        channel: Channel used for nucleus segmentation
        contour: Coordinates of the Nucleus contour.
        index: Index of the nucleus.
        label: Label of the Nucleus
    """
    field: Field
    channel: int
    contour: np.ndarray
    index: int = 0
    label: str = ""

    @property
    def centre(self) -> Tuple[int, int]:
        """
        Centroid of the Nucleus as row-major.
        """
        moments = cv2.moments(self.contour)
        centre_x = int(moments["m10"] / (moments["m00"] + 1e-5))
        centre_y = int(moments["m01"] / (moments["m00"] + 1e-5))

        return int(centre_y), int(centre_x)

    @property
    def centre_xy(self) -> Tuple[int, int]:
        """
        XY representation of the centre of the contour
        Returns:

        """
        y, x = self.centre

        return int(x), int(y)

    @property
    def intensity(self) -> int:
        _data = self.field.data[self.channel, ...]
        mask = np.zeros_like(_data)
        cv2.drawContours(mask, [self.contour], 0, (1,), -1)
        mask = mask > 0
        masked = _data.copy()
        masked[~mask] = 0

        return int(np.sum(masked))

    @property
    def area(self) -> int:
        return int(cv2.contourArea(self.contour))

    @property
    def full_in_field(self) -> bool:
        """
        Checks if a contour is fully visible.

        Specifically, it checks whether its centre is within the 5-% margin of the image.

        """
        h, w = self.field.data.shape[-2:]
        fraction = 0.05
        pad_lower = int(fraction * h)
        pad_upper = h - pad_lower
        centroid = self.centre

        if all([pad_lower < c < pad_upper for c in centroid]):
            return True

        return False

    def as_dict(self) -> dict:
        """
        Converts a Contour into a dictionary for serialisation.

        Returns: Dictionary with keys: channel, pos_r, pos_c, intensity, surface area, is_nucleus_full, contour.

        """

        result = {
            "channel": self.channel,
            "pos_r": self.centre[0],
            "pos_c": self.centre[1],
            "intensity": self.intensity,
            "surface_area": self.area,
            "is_nucleus_full": self.full_in_field,
            "contour": self.contour.tolist()
        }

        return result
