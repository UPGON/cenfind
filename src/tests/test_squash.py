from pathlib import Path

import numpy as np
import tifffile as tf

from centrack.commands.squash import project, write_projection


def random_data(shape):
    """
    Return random numpy array.
    from @cgohlke
    """
    return np.ones(shape, dtype='uint16')


def write_data(path_raw_file, data):
    with tf.TiffWriter(path_raw_file) as tif:
        tif.save(data,
                 photometric='minisblack',
                 metadata={'axes': 'CZYX',
                           })


# array_shape = (4, 67, 2048, 2048)
# field_synthetic = random_data(array_shape)

class TestSquash:
    path_data = Path(
        '/Users/leoburgy/Dropbox/epfl/projects/centrack/tests/data')
    path_raw_file = path_data / 'raw' / '20210727_RPE1_p53-Control_DAPI+rPOC5AF488+mHA568+gCPAP647_1_MMStack_Default.ome.tif'
    path_projections = path_data / 'projections'
    path_projections.mkdir(exist_ok=True)

    pixel_size, data_projected = project(path_raw_file)

    def test_project(self):
        assert self.data_projected.shape == (4, 2048, 2048)
        assert (self.pixel_size == .1025) or (self.pixel_size is None)

    def test_write_projection(self):
        write_projection(self.path_projections / f'{self.path_raw_file.stem}_max.tif', self.data_projected,
                         self.pixel_size)
