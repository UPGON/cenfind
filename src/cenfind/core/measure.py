from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from cenfind.core.data import Field
from cenfind.core.log import get_logger
from cenfind.core.outline import Point, Contour

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


def assign(nuclei: List[Contour], centrioles: List[Point], vicinity: float = 0) -> np.ndarray:
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

    num_nuclei = len(nuclei)
    num_centrioles = len(centrioles)

    costs = {}

    for i in range(num_nuclei):
        for j in range(num_centrioles):
            dist = signed_distance(centrioles[j], nuclei[i])
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


def score_nuclei(assigned: np.ndarray, nuclei: List[Contour], field_name: str, channel: int):
    """
    Score nuclei using the assignment matrix.
    Parameters
    ----------
    channel
    field_name
    assigned
    nuclei

    Returns
    -------

    """
    result = assigned.sum(axis=1)
    result = list(zip(nuclei, result))

    result = pd.DataFrame(list((n.index, s) for n, s in result))
    result.columns = ['nucleus', 'score']
    result["field"] = field_name
    result["channel"] = channel
    result = result.set_index(["field", "channel"])
    return result


def frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the absolute frequency of number of centriole per well or per field
    :param df: Df containing the number of centriole per nuclei
    :return: Df with absolute frequencies.

    Parameters
    ----------
    scored
    """
    cuts = [0, 1, 2, 3, 4, 5, np.inf]
    labels = "0 1 2 3 4 +".split(" ")

    result = pd.cut(df["score"], cuts, right=False, labels=labels,
                    include_lowest=True)
    result = result.groupby(["field", "channel"]).value_counts()
    result.name = "freq_abs"
    result = result.reset_index()
    result = result.rename({"score": "score_category"}, axis=1)
    result = result.pivot(index=["channel", "field"], columns="score_category")
    result.columns = labels

    return result


def assign_centrioles(assigned: np.ndarray, nuclei: List[Contour], centrioles: List[Point]) -> List[Tuple[int, int]]:
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


def proportion_cilia(field: Field, cilia: List[Point], nuclei: List[Contour], channel_cilia: int) -> pd.DataFrame:
    proportions = []
    ciliated = len(cilia) / len(nuclei)
    proportions.append({'field': field.name,
                        "channel_cilia": channel_cilia,
                        "n_nuclei": len(nuclei),
                        "n_cilia": len(cilia),
                        "p_ciliated": round(ciliated, 2)})
    return pd.DataFrame(proportions)
