from typing import List

import cv2
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from cenfind.core.log import get_logger
from cenfind.core.outline import Centre, Contour

logger = get_logger(__name__)


def signed_distance(focus: Centre, nucleus: Contour) -> float:
    """Wrapper for the opencv PolygonTest"""
    result = cv2.pointPolygonTest(nucleus.contour, focus.to_cv2(), measureDist=True)
    return result


def full_in_field(nucleus: Contour, image_shape, fraction) -> bool:
    h, w = image_shape
    pad_lower = int(fraction * h)
    pad_upper = h - pad_lower
    centroid = nucleus.centre.to_numpy()
    if all([pad_lower < c < pad_upper for c in centroid]):
        return True
    return False


def flag(is_full: bool) -> tuple:
    return (0, 255, 0) if is_full else (0, 0, 255)


def assign(
        nuclei: List[Contour], centrioles: List[Centre], vicinity=0
) -> List[Contour]:
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

    for i in range(num_nuclei):
        for j in range(num_centrioles):
            if x[i, j].solution_value() > 0.5:
                logger.debug(f"Adding Centre {j} to Nucleus {i}")
                _nuclei[i].add_centrioles(_centrioles[j])

    return _nuclei


def field_score_frequency(df, by="field"):
    """
    Count the absolute frequency of number of centriole per well or per field
    :param df: Df containing the number of centriole per nuclei
    :param by: the unit to pool nuclei, either `field` or `well`
    :return: Df with absolut frequencies.

    """
    cuts = [0, 1, 2, 3, 4, 5, np.inf]
    labels = "0 1 2 3 4 +".split(" ")

    df = df.set_index(["fov", "channel"])
    result = pd.cut(df["score"], cuts, right=False, labels=labels, include_lowest=True)
    result = result.groupby(["fov", "channel"]).value_counts()
    result.name = "freq_abs"
    result = result.sort_index().reset_index()
    result = result.rename({"score": "score_cat"}, axis=1)
    if by == "well":
        split_well = result["fov"].str.split("_", expand=True)
        if len(split_well.columns) != 2:
            raise ValueError("The name split has more than 2 parts (e.g. %s)" % result["fov"][0])
        result[["well", "field"]] = split_well
        print(result.columns)
        result = result.groupby(["well", "channel", "score_cat"])[["freq_abs"]].sum()
        result = result.reset_index()
        result = result.pivot(index=["well", "channel"], columns="score_cat")
        result.reset_index().sort_values(["channel", "well"])
    else:
        result = result.groupby(["fov", "channel", "score_cat"]).sum()
        result = result.reset_index()
        result = result.pivot(index=["fov", "channel"], columns="score_cat")
        result.reset_index().sort_values(["channel", "fov"])

    return result


def save_foci(foci_list: list[Centre], dst: str) -> None:
    if len(foci_list) == 0:
        array = np.array([])
        logger.info("No centriole detected")

    else:
        array = np.asarray(np.stack([c.to_numpy() for c in foci_list]))
        array = array[:, [1, 0]]
    np.savetxt(dst, array, delimiter=",", fmt="%u")
