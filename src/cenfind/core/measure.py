import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from spotipy.utils import points_matching

from cenfind.core.data import Dataset, Field
from cenfind.core.detectors import extract_foci
from cenfind.core.outline import Centre, Contour
from cenfind.core.log import get_logger

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
                logger.debug("Adding Centre %s to Nucleus %s" % (j, i))
                _nuclei[i].add_centrioles(_centrioles[j])

    return _nuclei


# TODO: refactor as a field method
def score(
        field,
        nuclei_scored,
        channel: int,
) -> List[dict]:
    """
    1. Detect foci in the given channels
    2. Detect nuclei
    3. Assign foci to nuclei
    :param field: The field to score
    :param nuclei_scored: the nuclei with the field centrioles filled
    :param channel:
    :return: list of scores
    """
    image_shape = field.projection.shape[1:]
    scores = []
    for nucleus in nuclei_scored:
        scores.append(
            {
                "fov": field.name,
                "channel": channel,
                "nucleus": nucleus.centre.to_numpy(),
                "score": len(nucleus.centrioles),
                "is_full": full_in_field(nucleus, image_shape, 0.05),
            }
        )
    return scores


def field_score_frequency(df, by="field"):
    """
    Count the absolute frequency of number of centriole per well or per field
    :param df: Df containing the number of centriole per nuclei
    :param by: the unit to group by, either `well` or `field`
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
        result[["well", "field"]] = result["fov"].str.split("_", expand=True)
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


def save_foci(foci_list: list[Centre], dst: str, logger=None) -> None:
    if len(foci_list) == 0:
        array = np.array([])
        if logger is not None:
            logger.info("No centriole detected")
        else:
            print("No centriole detected")
    else:
        array = np.asarray(np.stack([c.to_numpy() for c in foci_list]))
        array = array[:, [1, 0]]
    np.savetxt(dst, array, delimiter=",", fmt="%u")


# TODO: refactor as a field method
def field_metrics(
        field: Field,
        channel: int,
        annotation: np.ndarray,
        predictions: np.ndarray,
        tolerance: int,
        threshold: float,
) -> dict:
    """
    Compute the accuracy of the prediction on one field.
    :param field:
    :param channel:
    :param annotation:
    :param predictions:
    :param tolerance:
    :param threshold:
    :return: dictionary of metrics for the field
    """
    if all((len(predictions), len(annotation))) > 0:
        res = points_matching(annotation, predictions, cutoff_distance=tolerance)
    else:
        logging.warning(
            "threshold: %f; detected: %d; annotated: %d... Set precision and accuracy to zero"
            % (threshold, len(predictions), len(annotation))
        )
        res = SimpleNamespace()
        res.precision = 0.0
        res.recall = 0.0
        res.f1 = 0.0
        res.tp = (0,)
        res.fp = (0,)
        res.fn = 0
    perf = {
        "dataset": field.dataset.path.name,
        "field": field.name,
        "channel": channel,
        "n_actual": len(annotation),
        "n_preds": len(predictions),
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


# TODO: refactor as a field method
def dataset_metrics(
        dataset: Dataset, split: str, model: Path, tolerance, threshold
) -> list[dict]:
    """
    Apply field_metrics for every field of the dataset (test or train split)
    :param dataset
    :param split
    :param model,
    :param tolerance
    :param threshold
    :return list of metrics dictionaries
    """
    if type(tolerance) == int:
        tolerance = [tolerance]
    perfs = []
    for field, channel in dataset.pairs(split):
        annotation = field.annotation(channel)
        predictions = extract_foci(field, model, channel, prob_threshold=threshold)
        for tol in tolerance:
            perf = field_metrics(
                field, channel, annotation, predictions, tol, threshold=threshold
            )
            perfs.append(perf)
    return perfs
