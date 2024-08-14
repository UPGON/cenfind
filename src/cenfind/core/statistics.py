from typing import List
import logging

import numpy as np
import pandas as pd

from spotipy.utils import points_matching
from types import SimpleNamespace

from cenfind.core.data import Field
from cenfind.core.structures import Centriole, Nucleus

logger = logging.getLogger(__name__)


def evaluate(
        field: Field,
        channel: int,
        annotation: np.ndarray,
        predictions: List[Centriole],
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
    _predictions = [f.centre_xy for f in predictions]
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

def frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the absolute frequency of number of centriole per field.

    The bins are 0, 1, 2, 3, 4, +.

    Args:
        df: Dataframe with the number of centriole per nuclei

    Returns: Dataframe with absolute frequencies

    """
    cuts = [0, 1, 2, 3, 4, 5, np.inf]
    labels = "0 1 2 3 4 +".split(" ")

    df = df.loc[df["full_in_field"]]

    result = pd.cut(df["score"], cuts, right=False, labels=labels,
                    include_lowest=True)
    result = result.groupby(["field", "channel"]).value_counts()
    result.name = "freq_abs"
    result = result.reset_index()
    result = result.rename({"score": "score_category"}, axis=1)
    result = result.pivot(index=["channel", "field"], columns="score_category")
    result.columns = labels

    return result


def proportion_cilia(field: Field, cilia: List[Centriole], nuclei: List[Nucleus], channel_cilia: int) -> pd.DataFrame:
    """
    Computes the proportion of ciliated cells in one field of view.

    Args:
        field: The field of view analysed.
        cilia: List of detected cilia.
        nuclei: List of detected nuclei.
        channel_cilia: Channel used for cilia detection.

    Returns: Dataframe with field | channel cilia | n_nuclei | n_cilia | p_ciliated columns.

    """
    proportions = []
    ciliated = len(cilia) / len(nuclei)
    proportions.append({"field": field.name,
                        "channel_cilia": channel_cilia,
                        "n_nuclei": len(nuclei),
                        "n_cilia": len(cilia),
                        "p_ciliated": round(ciliated, 2)})
    return pd.DataFrame(proportions)
