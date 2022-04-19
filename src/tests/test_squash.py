import numpy as np
from pathlib import Path
from centrack.commands.squash import squash, read_ome_tif

def test_read_ome_tif():
    path = Path('../../data/RPE1wt_CEP152+GTU88+PCNT_1/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif')
    data = read_ome_tif(path)
    assert data.shape == (4, 67, 2048, 2048)

def test_squash():
    data = np.ndarray((4, 62, 2048, 2048))
    projected = squash(data)
    assert projected.shape == (4, 2048, 2048)
