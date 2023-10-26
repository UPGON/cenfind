import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tifffile as tif

from cenfind.core.log import get_logger
from cenfind.core.structures import Centriole, Nucleus

logger = get_logger(__name__)


def save_assigned(dst: Path, assigned: np.ndarray) -> None:
    """
    Saves the assignment matrix as a text file.

    Args:
        dst: The destination path
        assigned: the assignment matrix

    Returns:
    """

    logger.info("Writing assigned matrix to %s" % str(dst))
    np.savetxt(str(dst), assigned, fmt="%i")


def save_assigned_centrioles(dst: Path, assigned_centrioles) -> None:
    """
    Saves the assigned centrioles as a text file.

    The file is a mapping between the centriole and nucleus indices.

    Args:
        dst: The destination path
        assigned_centrioles: the assignment matrix

    Returns:
    """

    result = pd.DataFrame(assigned_centrioles)
    result.columns = ["centriole_index", "nucleus_index"]
    result.to_csv(dst, sep='\t', index=False)


def save_points(dst: Path, centrioles: List[Centriole]) -> None:
    """Saves the centrioles as a text file.

        The file contains the position of the centrioles in row-major format and signal intensity.

        Args:
           dst: The destination path.
           centrioles: the list of centrioles.

    Returns:
    """
    if len(centrioles) == 0:
        result = pd.DataFrame([])
        logger.info("No centriole detected")
    else:
        result = pd.DataFrame.from_dict({c.index: c.as_dict() for c in centrioles}, orient='index')
    result.to_csv(dst, index_label="index", index=False, sep='\t')


def save_contours(dst: Path, nuclei: List[Nucleus]) -> None:
    """
       Saves the centrioles as a text file.

       The file contains the position of the centrioles in row-major format and signal intensity.

       Args:
           dst: The destination path.
           nuclei: the list of centrioles.

       Returns:
    """
    with open(dst, "w") as file:
        json.dump({"nuclei": {nucleus.index: nucleus.as_dict() for nucleus in nuclei}}, file)
        logger.info("Writing contours to %s" % str(dst))


def save_visualisation(dst: Path, vis: np.ndarray) -> None:
    """
    Saves the visualisation image.

    Args:
       dst: The destination path.
       vis: The image.

    """
    logger.info("Writing visualisation to %s" % (str(dst)))
    tif.imwrite(dst, vis)
