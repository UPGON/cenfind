import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tifffile as tif

from cenfind.core.data import Field
from cenfind.core.log import get_logger
from cenfind.core.measure import full_in_field
from cenfind.core.structures import Point, Contour
from cenfind.core.visualisation import draw_contour, visualisation, create_vignette

logger = get_logger(__name__)


def save_assigned(dst: Path, assigned: np.ndarray) -> None:
    """
    Save the assignment matrix to a file
    Parameters
    ----------
    dst
    assigned

    Returns
    -------

    """
    logger.info("Writing assigned matrix to %s" % str(dst))
    np.savetxt(str(dst), assigned, fmt='%i')


def save_assigned_centrioles(dst: Path, assigned_centrioles):
    """
    Save assigned centrioles.
    Parameters
    ----------
    dst
    assigned_centrioles

    Returns
    -------

    """
    result = pd.DataFrame(assigned_centrioles)
    result.columns = ['centriole_index', 'nucleus_index']
    result.to_csv(dst, sep='\t', index=False)


def save_foci(dst: Path, centrioles: List[Point], image: np.ndarray) -> None:
    if len(centrioles) == 0:
        result = pd.DataFrame([])
        logger.info("No centriole detected")
    else:
        container = []
        for c in centrioles:
            rec = {"index": c.index,
                   "channel": c.channel,
                   "pos_r": c.centre[0],
                   "pos_c": c.centre[1],
                   "intensity": c.intensity(image)}
            container.append(rec)
        result = pd.DataFrame(container)
    result.to_csv(dst, index_label='index', index=False, sep='\t')


def save_nuclei_mask(dst: Path, nuclei: List[Contour], image):
    """
    Save the detected nuclei as a mask.
    Parameters
    ----------
    dst
    nuclei
    image

    Returns
    -------

    """
    result = np.zeros_like(image, dtype='uint8')
    for nucleus in nuclei:
        result = draw_contour(result, nucleus, color=255, annotation=False, thickness=-1)
    cv2.imwrite(str(dst), result)


def save_nuclei_contour(dst: Path, nuclei: List[Contour]):
    container = {}
    for nucleus in nuclei:
        container[nucleus.index] = nucleus.contour.tolist()
    with open(dst, 'w') as file:
        json.dump(container, file)
        logger.info('Writing contours to %s' % str(dst))


def save_nuclei(dst: Path, nuclei: List[Contour], image):
    """
    Save nuclei as a table with measurements and position.
    Parameters
    ----------
    dst
    nuclei
    image

    Returns
    -------

    """
    container = []
    for nucleus in nuclei:
        rec = {"index": nucleus.index,
               "channel": nucleus.channel,
               "pos_r": nucleus.centre.centre[0],
               "pos_c": nucleus.centre.centre[1],
               "intensity": nucleus.intensity(image),
               "surface_area": nucleus.area(),
               "is_nucleus_full": full_in_field(nucleus, image.shape, 0.05),
               }
        container.append(rec)
    result = pd.DataFrame(container)
    result.to_csv(dst, sep='\t')


def save_visualisation(dst, field: Field,
                       channel_centrioles: int,
                       channel_nuclei: int,
                       centrioles: List[Point] = None,
                       nuclei: List[Contour] = None,
                       assigned: List[Tuple[Point, Contour]] = None) -> None:
    background = create_vignette(field, marker_index=channel_centrioles, nuclei_index=channel_nuclei)
    vis = visualisation(background, centrioles=centrioles, nuclei=nuclei, assigned=assigned)
    logger.info("Writing visualisation to %s" % (str(dst)))
    tif.imwrite(dst, vis)
