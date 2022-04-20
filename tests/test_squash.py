import os
import tempfile

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
    collect_ome_tif, write_projection,
    )

HERE = Path(__file__).resolve()
TEMP_DIR = HERE / '_tmp'
if not TEMP_DIR.exists():
    TEMP_DIR = tempfile.gettempdir()


def random_data(shape, dtype):
    rng = np.random.RandomState(1993)
    dtype_info = np.iinfo(dtype)
    return rng.randint(low=dtype_info.min,
                       high=dtype_info.max,
                       size=shape,
                       dtype=dtype)


class TempFileName:
    """
    Temporary file name context manager.
    from @cgholke
    """
    def __init__(self, name=None, ext='.tif', remove=False):
        self.remove = remove or TEMP_DIR == tempfile.gettempdir()
        if not name:
            fh = tempfile.NamedTemporaryFile(prefix='test_')
            self.name = fh.named
            fh.close()
        else:
            self.name = TEMP_DIR / f'test_{name}{ext}'

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remove:
            try:
                os.remove(self.name)
            except OSError:
                pass



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
        _plane[30, 1, r, c] = 1

    return _plane


@pytest.fixture(scope='class')
def path_dataset():
    return Path('data')


@pytest.fixture(scope='class')
def path_ome():
    return Path(
        'data/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000.ome.tif')


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


class TestOMETIFF:
    def test_extract_pixel_size(self, path_ome):
        pixel_size = extract_pixel_size(path_ome)
        assert pixel_size == 1.025e-05

    def test_extract_axes_order(self, path_ome):
        order = extract_axes_order(path_ome)
        assert order in ('CZYX', 'ZCYX')

    def test_read_ome_tif(self, path_ome):
        data = load_ome_tif(path_ome)
        assert data.shape == (4, 67, 2048, 2048)

    def test_read_ome(self, path_ome):
        pixel_size, data = read_ome_tif(path_ome)
        assert pixel_size == 1.025e-5
        assert data.shape == (4, 67, 2048, 2048)


class TestDataset:
    def test_collect_ome_tif(self, path_dataset):
        files_to_process = collect_ome_tif(path_dataset)
        assert all(
            [f.name.endswith('.ome.tif') for f in files_to_process]) == True

    def test_squash_dataset(self, path_dataset):
        files_to_process = collect_ome_tif(path_dataset)
        path_projections = path_dataset / 'projections'
        for f in files_to_process:
            pixel_size, data = read_ome_tif(f)
            projected = squash(data)
            file_name = f"{f.stem.removesuffix('.ome.tif')}_max.tif"
            write_projection(path_projections / file_name, projected,
                             pixel_size=pixel_size)

        assert all(
            [f.name.endswith('_max.tif') for f in path_projections.iterdir()])
