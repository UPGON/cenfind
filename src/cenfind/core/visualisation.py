from typing import List

import cv2
import numpy as np
from skimage.draw import disk
from skimage.exposure import rescale_intensity

from cenfind.core.data import Field
from cenfind.core.log import get_logger
from cenfind.core.structures import Centriole, Nucleus

logger = get_logger(__name__)


def resize_image(image: np.ndarray, factor: int = 256) -> np.ndarray:
    """
    Resizes the image for nuclei segmentation by StarDist

    Args:
        image: a single channel image (H x W)
        factor: the target dimension

    Returns: Resized image

    """

    height, width = image.shape
    shrinkage_factor = int(height // factor)
    height_scaled = int(height // shrinkage_factor)
    width_scaled = int(width // shrinkage_factor)
    data_resized = cv2.resize(image,
                              dsize=(height_scaled, width_scaled),
                              fx=1, fy=1, interpolation=cv2.INTER_NEAREST,
                              )

    return data_resized


def draw_point(image: np.ndarray, point: Centriole,
               color: tuple[int, int, int] = (0, 255, 0), annotation: bool = True,
               marker_type: int = cv2.MARKER_SQUARE, marker_size: int = 8,
               ):
    """
    Draws a point object on an image.

    Args:
        image: Image to draw the point on.
        point: The point to draw
        color: The color to use (default: Green)
        annotation: whether to add text.
        marker_type: Marker type for point (default: square)
        marker_size: Marker size (default: 8 px)

    Returns: Annotated image.

    """
    r, c = point.centre
    offset_col = int(0.01 * image.shape[1])

    if annotation:
        cv2.putText(
            image,
            f"{point.label} {point.index}",
            org=(c + offset_col, r),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            thickness=1,
            color=color,
        )

    return cv2.drawMarker(
        image, (c, r), color, markerType=marker_type, markerSize=marker_size
    )


def draw_foci(image: np.ndarray, foci: list[Centriole], radius=2) -> np.ndarray:
    """
    Draws foci as disks of given radius.

    Used to group centrioles into centrosomes.

    Args:
        image: Image to draw foci on
        foci: List of centrioles to draw.
        radius: Radius of the centriole as disk

    Returns: The mask image

    """

    mask = np.zeros(image.shape, dtype="uint8")
    for f in foci:
        rr, cc = disk(f.centre, radius, shape=image.shape[-2:])
        mask[rr, cc] = 250

    return mask


def draw_contour(image: np.ndarray, nucleus: Nucleus,
                 color: tuple[int, int, int] = (0, 255, 0), annotation: bool = True, thickness: int = 2):
    """
    Draws a contour from a Nucleus object on an image.

    Args:
        image: Image to draw on the contour
        nucleus: Contour to draw
        color: colour as tuple
        annotation: if True, add text to object
        thickness: Thickness of the contour line

    Returns: the masked image.

    """
    r, c = nucleus.centre
    cv2.drawContours(image, [nucleus.contour], -1, color, thickness=thickness)
    if annotation:
        cv2.putText(
            image,
            f"{nucleus.label}{nucleus.index}",
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


def _color_channel(image: np.ndarray, color: tuple[int, int, int], out_range: str) -> np.ndarray:
    """
    Creates a coloured version of a channel image

    Args:
        image: Image to convert
        color: Color to use (B, G, R)
        out_range: Depth, e.g., 'uint8'

    Returns:

    """
    """

    :param data: the data to use
    :param color: the color as a tuple (B, G, R)
    :param out_range:
    :return:
    """
    data = rescale_intensity(image, out_range=out_range)
    b = np.multiply(data, color[0], casting="unsafe")
    g = np.multiply(data, color[1], casting="unsafe")
    r = np.multiply(data, color[2], casting="unsafe")
    res = cv2.merge([b, g, r])

    return res


def create_vignette(field: Field, marker_index: int, nuclei_index: int) -> np.ndarray:
    """
    Normalises all markers and represent them as blue and highlight the channel in green.

    Args:
        field: Field to use
        marker_index: Channel for th marker
        nuclei_index: Channel

    Returns:

    """

    layer_nuclei = field.data[nuclei_index, ...]
    layer_marker = field.data[marker_index, ...]

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


def visualisation(background: np.ndarray,
                  centrioles: List[Centriole],
                  nuclei: List[Nucleus],
                  assigned: np.ndarray = None,
                  ) -> np.ndarray:
    """
    Builds a visualisation image to assess the quality of the result.

    Args:
        background: The image to draw on.
        centrioles: Centrioles to be drawn.
        nuclei: Nuclei to be drawn.
        assigned: Arrows to draw between centrioles and nuclei.

    Returns: Annotated image

    """
    for centriole in centrioles:
        background = draw_point(background, centriole, annotation=False)

    for nucleus in nuclei:
        if nucleus.full_in_field:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        background = draw_contour(background, nucleus, color=color, annotation=False)

    if assigned is None:
        return background

    if not centrioles or not nuclei:
        return background

    it = np.nditer(assigned, flags=['multi_index'])
    for entry in it:
        if not entry:
            continue
        n, c = it.multi_index
        nucleus = nuclei[n]
        centriole = centrioles[c]
        cv2.arrowedLine(background, tuple(reversed(centriole.centre)), tuple(reversed(nucleus.centre)),
                        color=(0, 255, 0), thickness=2)

    return background
