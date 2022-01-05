import numpy
import numpy as np
import tifffile as tf

from scripts.simple_centrack import (
    load_ome,
    project,
    detect_foci,
    segment_nuclei,
    assign
)


def random_data(shape):
    """
    Return random numpy array.
    from @cgohlke
    """
    return numpy.ones(shape, dtype='uint16')


def write_data(dst, data):
    with tf.TiffWriter(dst) as tif:
        tif.save(data,
                 photometric='minisblack',
                 metadata={'axes': 'CZYX',
                           })


def test_load_ome(tmp_path):
    tmp_path_data = tmp_path / 'temp.ome.tif'

    array_shape = (4, 67, 2048, 2048)
    field_synthetic = random_data(array_shape)
    write_data(tmp_path_data, field_synthetic)

    pixel_size, data = load_ome(tmp_path_data)
    assert data.shape == array_shape
