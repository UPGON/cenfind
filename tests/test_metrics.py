import numpy as np

from centrack.metrics import compute_metrics


def test_compute_metrics():
    positions = np.array([
        [2, 2],
        [2, 5],
        [2, 7],
        [6, 2],
        [6, 5],
        [8, 2]
    ])

    predictions = np.array([
        [2, 2],
        [2, 6],
        [5, 3],
        [8, 2],
        [8, 8]
    ])

    result = compute_metrics(positions, predictions, offset_max=2)

    assert result == {'fp': {4},
                      'fn': {2, 4},
                      'tp': {0, 1, 2, 3}
                      }
