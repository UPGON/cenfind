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
    logger.info("Writing assigned matrix to %s" % str(dst))
    np.savetxt(str(dst), assigned, fmt='%i')


def save_assigned_centrioles(dst: Path, assigned_centrioles) -> None:
    result = pd.DataFrame(assigned_centrioles)
    result.columns = ['centriole_index', 'nucleus_index']
    result.to_csv(dst, sep='\t', index=False)


def save_points(dst: Path, centrioles: List[Centriole]) -> None:
    if len(centrioles) == 0:
        result = pd.DataFrame([])
        logger.info("No centriole detected")
    else:
        result = pd.DataFrame.from_dict({c.index: c.as_dict() for c in centrioles}, orient='index')
    result.to_csv(dst, index_label='index', index=False, sep='\t')


def save_contours(dst: Path, nuclei: List[Nucleus]) -> None:
    with open(dst, 'w') as file:
        json.dump({nucleus.index: nucleus.as_dict() for nucleus in nuclei}, file)
        logger.info('Writing contours to %s' % str(dst))


def save_visualisation(dst, vis: np.ndarray) -> None:
    logger.info("Writing visualisation to %s" % (str(dst)))
    tif.imwrite(dst, vis)
