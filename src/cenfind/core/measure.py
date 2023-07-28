from pathlib import Path
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from spotipy.utils import points_matching

from cenfind.core.data import Field, Dataset
from cenfind.core.log import get_logger
from cenfind.core.outline import Point, Contour, draw_contour

logger = get_logger(__name__)


def full_in_field(nucleus: Contour, image_shape, fraction) -> bool:
    """
    Check if a contour is fully visible.
    Parameters
    ----------
    nucleus
    image_shape
    fraction

    Returns
    -------

    """
    h, w = image_shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    centroid = nucleus.centre.to_numpy()
    if all([pad_lower < c < pad_upper for c in centroid]):
        return True
    return False


def flag(is_full: bool) -> tuple:
    """
    Helper function to change colour of contour.
    Parameters
    ----------
    is_full

    Returns
    -------

    """
    return (0, 255, 0) if is_full else (0, 0, 255)


def signed_distance(focus: Point, nucleus: Contour) -> float:
    """
    Wrapper for the opencv PolygonTest
    """

    result = cv2.pointPolygonTest(nucleus.contour, focus.to_cv2(), measureDist=True)
    return result


def assign(nuclei: List[Contour], centrioles: List[Point], vicinity=0) -> np.ndarray:
    """
    Solve the linear assignment of n centrioles nearest to 1 nucleus, up to a threshold.
    Parameters
    ----------
    nuclei
    centrioles
    vicinity

    Returns
    -------

    """
    _nuclei = nuclei.copy()
    _centrioles = centrioles.copy()

    num_nuclei = len(_nuclei)
    num_centrioles = len(_centrioles)

    costs = {}

    for i in range(num_nuclei):
        for j in range(num_centrioles):
            dist = signed_distance(_centrioles[j], _nuclei[i])
            costs[i, j] = dist + vicinity
    solver = pywraplp.Solver.CreateSolver("SCIP")

    x = {}
    for i in range(num_nuclei):
        for j in range(num_centrioles):
            x[i, j] = solver.IntVar(0, 1, "")

    for j in range(num_centrioles):
        solver.Add(solver.Sum([x[i, j] for i in range(num_nuclei)]) <= 1)

    # Objective
    objective_terms = []
    for i in range(num_nuclei):
        for j in range(num_centrioles):
            objective_terms.append(costs[i, j] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        raise ValueError("No solution found.")

    result = np.zeros([num_nuclei, num_centrioles], dtype=bool)
    for i in range(num_nuclei):
        for j in range(num_centrioles):
            if x[i, j].solution_value() > 0:
                result[i, j] = True

    return result


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


def save_nuclei_mask(path: Path, nuclei: List[Contour], image):
    """
    Save the detected nuclei as a mask.
    Parameters
    ----------
    path
    nuclei
    image

    Returns
    -------

    """
    result = np.zeros_like(image, dtype='uint8')
    for nucleus in nuclei:
        result = draw_contour(result, nucleus, color=255, annotation=False, thickness=-1)
    cv2.imwrite(str(path), result)


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


def score_nuclei(assigned, nuclei):
    """
    Score nuclei using the assignment matrix.
    Parameters
    ----------
    assigned
    nuclei

    Returns
    -------

    """
    scores = assigned.sum(axis=1)
    return list(zip(nuclei, scores))


def save_scores(dst, scores):
    """
    Save scores produced by score_nuclei.
    Parameters
    ----------
    dst
    scores

    Returns
    -------

    """
    result = pd.DataFrame(list((n.index, s) for n, s in scores))
    result.columns = ['nuclei_index', 'centriole_number']
    result.to_csv(dst, sep='\t', index=False)


def assign_centrioles(assigned, nuclei, centrioles):
    """
    Assign nucleus index to centrioles index, or -1 if no nucleus.
    Parameters
    ----------
    assigned
    nuclei
    centrioles

    Returns
    -------

    """
    result = []
    for c, centriole in enumerate(assigned.T):
        centriole_index = centrioles[c].index
        if centriole.max() == 0:
            result.append((centriole_index, -1))
        else:
            nucleus_matrix_index = np.where(centriole == 1)[0][0]
            nucleus_index = nuclei[nucleus_matrix_index].index
            result.append((centriole_index, nucleus_index))

    return result


def save_assigned_centrioles(dst, assigned_centrioles):
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


def evaluate(
        dataset: Dataset,
        field: Field,
        channel: int,
        annotation: np.ndarray,
        predictions: np.ndarray,
        tolerance: int,
        threshold: float,
) -> dict:
    """
    Compute the accuracy of the prediction on one field.
    :param dataset:
    :param field:
    :param channel:
    :param annotation:
    :param predictions:
    :param tolerance:
    :param threshold:
    :return: dictionary of metrics for the field
    """
    _predictions = [f.centre for f in predictions]
    if all((len(_predictions), len(annotation))) > 0:
        res = points_matching(annotation, _predictions, cutoff_distance=tolerance)
    else:
        logger.warning(
            "threshold: %f; detected: %d; annotated: %d... Set precision and accuracy to zero"
            % (threshold, len(_predictions), len(annotation))
        )
        res = SimpleNamespace()
        res.precision = 0.0
        res.recall = 0.0
        res.f1 = 0.0
        res.tp = (0,)
        res.fp = (0,)
        res.fn = 0
    perf = {
        "dataset": dataset.path.name,
        "field": field.name,
        "channel": channel,
        "n_actual": len(annotation),
        "n_preds": len(_predictions),
        "threshold": threshold,
        "tolerance": tolerance,
        "tp": res.tp[0] if type(res.tp) == tuple else res.tp,
        "fp": res.fp[0] if type(res.fp) == tuple else res.fp,
        "fn": res.fn[0] if type(res.fn) == tuple else res.fn,
        "precision": np.round(res.precision, 3),
        "recall": np.round(res.recall, 3),
        "f1": np.round(res.f1, 3),
    }
    return perf
