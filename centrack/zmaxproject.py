# Z-max projection
import logging
import re
from pathlib import Path

import numpy as np
import tifffile as tf
from tqdm import tqdm

from describe import PixelSize
from fetch import DataSet, Condition, is_tif

if __name__ == '__main__':
    dataset_path = Path(
        '/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1/raw')
    markers = 'DAPI+CEP152+GTU88+PCNT'.split('+')
    conditions = Condition(markers=markers,
                           genotype='RPE1wt',
                           pixel_size=PixelSize(.1025, 'um'))
    dataset = DataSet(dataset_path)
    dataset.projections.mkdir(exist_ok=True)

    files = [file for file in dataset.projections.iterdir() if is_tif(file)]
    logging.info('there are %d ome files', len([f for f in files]))
    pbar = tqdm(files)

    for file in pbar:
        dest_path = file.parent

        with tf.TiffFile(file) as field:
            metadata = field.ome_metadata
            order = field.series[0].axes
            logging.info('%s %s', file.name, order)
            data = field.asarray()

        if len(data.shape) < 5:
            data = np.expand_dims(data, 0)

        projected = data.max(axis=2)

        file_name = file.name
        file_name = file_name.removesuffix(''.join(file.suffixes))
        file_name = file_name.replace('', '')
        file_name = re.sub(r'_(Default|MMStack)_\d-Pos', '', file_name)
        file_name = file_name.replace('', '')

        dest_name = f'{file_name}_max.ome.tif'
        path_projected = dest_path / dest_name
        if path_projected.is_file():
            logging.warning('File exists: %s', path_projected)
            continue
        tf.imwrite(path_projected, projected, photometric='minisblack')
