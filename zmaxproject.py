# Z-max projection

from pathlib import Path
import re

import numpy as np
import tifffile as tf
from tqdm import tqdm

from centrack.data import DataSet, Field


if __name__ == '__main__':
    dataset_input = input('Enter the full path to the dataset folder: ')
    dataset_input = Path(dataset_input)
    if not dataset_input.exists():
        raise FileExistsError

    dataset = DataSet(dataset_input)
    dataset.projections.mkdir(exist_ok=True)

    files = dataset.fields

    pbar = tqdm(files)

    for file in pbar:

        with tf.TiffFile(file) as field:
            metadata = field.ome_metadata
            order = field.series[0].axes
            data = field.asarray()

        if order == 'ZCYX':
            z, c, y, x = data.shape
            data = data.reshape((c, z, y, x))

        if len(data.shape) < 5:
            data = np.expand_dims(data, 0)

        projected = data.max(axis=2)

        file_name = file.name
        file_name = file_name.removesuffix(''.join(file.suffixes))
        file_name = file_name.replace('', '')
        file_name = file_name.sub(r'_(Default|MMStack)_\d-Pos', '', file_name)
        file_name = file_name.replace('', '')

        dest_name = f'{file_name}_max.ome.tif'
        path_projected = dataset.projections / dest_name
        tf.imwrite(path_projected, projected, photometric='minisblack')
