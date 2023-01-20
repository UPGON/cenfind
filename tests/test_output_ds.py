from pathlib import Path
from cenfind.core.measure import field_score_frequency
import pandas as pd


def test_statistics():
    path_scores = Path('/home/buergy/UPGON/GG/siRNA_screen/BSF/InCellAnalyzer/DATA/20230106/Cenfind_20230119/20230106_60X_Grenier_RPE_BSF020217R1')
    scores = pd.read_csv(path_scores / 'statistics/scores_df.tsv', sep='\t')
    binned = field_score_frequency(scores)
