import numpy as np
import pytest
import tifffile as tf

from src.centrack.squash import (
    squash,
    read_ome_tif,
    correct_axes,
    extract_pixel_size,
    extract_axes_order,
)


@pytest.fixture(scope='session')
def stack():
    return np.zeros((4, 32, 2048, 2048),
                    dtype=np.uint16)


@pytest.fixture(scope='session')
def foci():
    return np.asarray([
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 3, 4],
        [0, 1, 400, 260],
        [0, 2, 1209, 3],
        [0, 3, 2000, 1500],
        [1, 3, 2000, 1500],
    ])


@pytest.fixture(scope='session')
def stack_with_foci(stack, foci):
    _plane = stack.copy()
    for c, z, h, w in foci:
        _plane[c, z, h, w] = 1
    return _plane


@pytest.fixture(scope='session')
def stack_ome_tif(tmp_path_factory, stack_with_foci):
    """
    Write a synthetic OME tiff file
    :param tmp_path_factory:
    :param stack_with_foci
    :return:
    """
    fn = tmp_path_factory.mktemp('data') / 'stack.ome.tif'
    with tf.TiffWriter(fn) as tif:
        tif.write(stack_with_foci, photometric='minisblack',
                  metadata={'axes': 'CZYX',
                            'resolution': 4.2}
                  )

    return fn


class TestOMETIFF:
    def test_squash(self, stack):
        projected = squash(stack)
        assert projected.shape == (4, 2048, 2048)

    def test_correct_axes(self, stack):
        swapped = correct_axes(stack)
        assert stack.shape == (4, 32, 2048, 2048)
        assert swapped.shape == (32, 4, 2048, 2048)

    def test_extract_pixel_size(self, stack_ome_tif):
        pixel_size = extract_pixel_size(stack_ome_tif)
        assert pixel_size == -1

    def test_extract_axes_order(self, stack_ome_tif):
        order = extract_axes_order(stack_ome_tif)
        assert order in ('CZYX', 'ZCYX')

    def test_read_ome_tif(self, stack_ome_tif):
        pixel_size, loaded = read_ome_tif(stack_ome_tif)
        assert pixel_size == -1
        assert loaded.shape == (4, 32, 2048, 2048)

    def test_correct_axes_values(self, stack_ome_tif):
        _, loaded = read_ome_tif(stack_ome_tif)
        swapped = correct_axes(loaded)
        swapped_projected_z = squash(swapped)
        assert loaded.max() == 1
        assert swapped_projected_z.max() == 1
