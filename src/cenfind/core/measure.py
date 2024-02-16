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
    Sets the colour of contour depending on whether it is full (green) or not (red).

    :param is_full:
    :return: Whether inside or outside
    """
    return (0, 255, 0) if is_full else (0, 0, 255)


def signed_distance(focus: Centriole, nucleus: Nucleus) -> float:
    """
    Wrapper for the opencv PolygonTest.

    :param focus: The point to test.
    :param nucleus: Reference nucleus.
    :return: Distance in pixel.
    """

    result = cv2.pointPolygonTest(nucleus.contour, focus.centre_xy, measureDist=True)
    return result


@define
class Assigner:
    """
    Computes, saves and manipulates the assignment matrix.

    Attributes:
        centrioles: List of centrioles.
        nuclei: List of nuclei.
        vicinity: Threshold distance to assign centrioles to nuclei.
        assignment: Numpy matrix that is filled by self._compute.
    """

    centrioles: List[Centriole]
    nuclei: List[Nucleus]
    vicinity: float = 0
    assignment: np.ndarray = None

    def _compute(self, vicinity: float = 0) -> np.ndarray:
        """
        Computes the assignment using the Google OR-Tool solver
        for the linear assignment with multiple tasks.
        Args:
            vicinity: Threshold distance to assign centrioles to nuclei.

        Returns: Assignment matrix (Nuclei x Centrioles)

        """
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
        """
        Scores nuclei for centrioles by summing over the columns of the assignment matrix.

        Args:
            field_name: Field name used for the result table
            channel: Channel used for the result table.

        Returns: DataFrame with the field name and the channel used as index and the nucleus id, the score.

        """
        if self.assignment is None:
            self.assignment = self._compute(self.vicinity)

        result = self.assignment.sum(axis=1)
        result = list(zip(self.nuclei, result))

        result = pd.DataFrame(list((n.index, n.full_in_field, s) for n, s in result))
        result.columns = ["nucleus", "full_in_field", "score"]
        result["field"] = field_name
        result["channel"] = channel
        result = result.set_index(["field", "channel"])
        return result

    def assign_centrioles(self) -> List[Tuple[int, int]]:
        """
        Assigns centrioles by looking up the nucleus id of the assignment matrix.

        If the centriole is not assigned, it is flagged with -1.

        Returns: List of tuples (Centriole ID, Nucleus ID)

        """
        if self.assignment is None:
            self.assignment = self._compute(self.vicinity)

        result = []

        for c, centriole in enumerate(self.assignment.T):
            centriole_index = self.centrioles[c].index
            if np.max(centriole) == 0:
                result.append((centriole_index, -1))
            else:
                nucleus_matrix_index = np.where(centriole == 1)[0][0]
                nucleus_index = self.nuclei[nucleus_matrix_index].index
                result.append((centriole_index, nucleus_index))

        return result
