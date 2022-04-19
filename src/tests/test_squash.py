import pytest
import numpy as np
from centrack.commands.squash import squash

def test_squash():
    data = np.ndarray((4, 62, 2048, 2048))
    projected = squash(data)
    assert projected.shape == (4, 2048, 2048)
