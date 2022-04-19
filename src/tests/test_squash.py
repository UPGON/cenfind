import numpy as np
from pathlib import Path

import pytest

from centrack.commands.squash import squash, read_ome_tif, correct_axes


@pytest.fixture()
def empty_stack_czyx():
    return np.zeros((4, 67, 2048, 2048), dtype='uint16')


@pytest.fixture()
def empty_stack_zcyx():
    return np.zeros((67, 4, 2048, 2048), dtype='uint16')


@pytest.fixture()
def projection():
    return np.zeros((4, 2048, 2048), dtype='uint16')


@pytest.fixture()
def foci():
    return np.asarray([
        [4, 26],
        [5, 20],
        [400, 260],
        [1209, 3],
        [2000, 1500],
    ])


@pytest.fixture()
def foci_mask(empty_stack_zcyx, foci):
    _plane = empty_stack_zcyx.copy()
    for r, c in foci:
        _plane[30, 1, r, c] = 1

    return _plane


def test_read_ome_tif():
    path = Path('../../data/RPE1wt_CEP152+GTU88+PCNT_1/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif')
    data = read_ome_tif(path)
    assert data.shape == (4, 67, 2048, 2048)


def test_squash(empty_stack_czyx, projection):
    data = empty_stack_czyx
    projected = squash(data)
    assert projection.shape == projected.shape


def test_correct_axes(empty_stack_zcyx, empty_stack_czyx):
    data = empty_stack_zcyx
    swapped = correct_axes(data)
    assert swapped.shape == empty_stack_czyx.shape


def test_correct_axes_values(foci_mask):
    projected = squash(foci_mask)
    swapped = correct_axes(foci_mask)
    swapped_projected = squash(swapped)
    assert foci_mask.sum() == 5
    assert projected.sum() == 5
    assert swapped_projected.sum() == 5
