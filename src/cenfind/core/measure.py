from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
from attrs import define
from ortools.linear_solver import pywraplp

from cenfind.core.log import get_logger
from cenfind.core.structures import Centriole, Nucleus

logger = get_logger(__name__)


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


def signed_distance(focus: Centriole, nucleus: Nucleus) -> float:
    """
    Wrapper for the opencv PolygonTest
    """

    result = cv2.pointPolygonTest(nucleus.contour, focus.centre_xy, measureDist=True)
    return result


@define
class Assigner:
    centrioles: List[Centriole]
    nuclei: List[Nucleus]
    vicinity: float = 0
    assignment: np.ndarray = None

    def _compute(self, vicinity: float = 0) -> np.ndarray:
        num_nuclei = len(self.nuclei)
        num_centrioles = len(self.centrioles)

        costs = {}
        for i in range(num_nuclei):
            for j in range(num_centrioles):
                dist = signed_distance(self.centrioles[j], self.nuclei[i])
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

    def score_nuclei(self, field_name: str, channel: int) -> pd.DataFrame:
        if self.assignment is None:
            self.assignment = self._compute(self.vicinity)

        result = self.assignment.sum(axis=1)
        result = list(zip(self.nuclei, result))

        result = pd.DataFrame(list((n.index, s) for n, s in result))
        result.columns = ['nucleus', 'score']
        result["field"] = field_name
        result["channel"] = channel
        result = result.set_index(["field", "channel"])
        return result

    def assign_centrioles(self) -> List[Tuple[int, int]]:
        if self.assignment is None:
            self.assignment = self._compute(self.vicinity)

        result = []
        for c, centriole in enumerate(self.assignment.T):
            centriole_index = self.centrioles[c].index
            if centriole.max() == 0:
                result.append((centriole_index, -1))
            else:
                nucleus_matrix_index = np.where(centriole == 1)[0][0]
                nucleus_index = self.nuclei[nucleus_matrix_index].index
                result.append((centriole_index, nucleus_index))

        return result
