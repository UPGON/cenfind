import numpy as np
from pathlib import Path

import pytest

from src.centrack.commands.squash import (
    squash,
    load_ome_tif,
    read_ome_tif,
    correct_axes,
    extract_pixel_size,
    extract_axes_order,
)


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
        [0, 0],
        [3, 0],
        [3, 4],
        [400, 260],
        [1209, 3],
        [2000, 1500],
    ])


@pytest.fixture()
def foci_mask(empty_stack_zcyx, foci):
    _plane = empty_stack_zcyx.copy()
    for i, (r, c) in enumerate(foci):
        _plane[i, 1, r, c] = 1

    return _plane


@pytest.fixture()
def path_ome():
    return Path('../../data/RPE1wt_CEP152+GTU88+PCNT_1/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif')


def test_extract_pixel_size(path_ome):
    pixel_size = extract_pixel_size(path_ome)
    assert pixel_size == 1.025e-05


def test_extract_axes_order(path_ome):
    order = extract_axes_order(path_ome)
    assert order in ('CZYX', 'ZCYX')


def test_read_ome_tif(path_ome):
    data = load_ome_tif(path_ome)
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
    swapped = correct_axes(foci_mask)
    projected = squash(foci_mask)
    swapped_projected = squash(swapped)
    assert foci_mask.sum() == 6
    assert projected.max() == 1
    assert swapped_projected.max() == 1


def test_read_ome(path_ome):
    pixel_size, data = read_ome_tif(path_ome)
    assert pixel_size == 1.025e-5
    assert data.shape == (4, 67, 2048, 2048)
