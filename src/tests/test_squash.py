import numpy as np
from pathlib import Path

import pytest

from centrack.commands.squash import squash, read_ome_tif

@pytest.fixture()
def empty_stack():
    return np.zeros((4, 67, 2048, 2048), dtype='uint16')

@pytest.fixture()
def projection():
    return np.zeros((4, 2048, 2048), dtype='uint16')

def test_read_ome_tif():
    path = Path('../../data/RPE1wt_CEP152+GTU88+PCNT_1/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif')
    data = read_ome_tif(path)
    assert data.shape == (4, 67, 2048, 2048)

def test_squash(empty_stack, projection):
    data = empty_stack
    projected = squash(data)
    assert projection.shape == projected.shape
