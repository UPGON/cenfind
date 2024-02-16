from typing import List

import numpy as np
import pandas as pd

from cenfind.core.data import Field
from cenfind.core.log import get_logger
from cenfind.core.structures import Centriole, Nucleus

logger = get_logger(__name__)


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
