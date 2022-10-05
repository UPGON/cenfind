import pandas as pd
from cenfind.core.measure import field_score_frequency

def test_field_score():
    print(9)
    scores = pd.read_csv('scores/scores_df.tsv', sep='\t')
    binned = field_score_frequency(scores)

    print(0)
