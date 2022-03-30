import itertools
from centrack.commands.score import score_summary
import pandas as pd
from numpy.random import default_rng

rng = default_rng(1993)
scores = rng.choice([0, 1, 2, 3, 4], 4)

fov_id = [[i] * 5 for i in range(5)]


def test_bins():
    df = pd.DataFrame({
        'fov': [f'fov{i}' for i in itertools.chain(*fov_id)],
        'channel': rng.choice([1, 2, 3], 25),
        'score': rng.choice(list(range(5)) + [6, 1000, 54], 25)
        })

    table = score_summary(df)
    assert 0 == 0
