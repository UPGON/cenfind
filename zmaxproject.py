# Z-max projection

import numpy as np
import tifffile as tf
from tqdm import tqdm
from aicsimageio import AICSImage

from centrack.data import DataSet, Field


if __name__ == '__main__':

    dataset = DataSet('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1')
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

        file_name_core = file.name.removesuffix(''.join(file.suffixes))

        dest_name = f'{file_name_core}_max.ome.tif'
        path_projected = dataset.projections / dest_name
        AICSImage(projected).save(path_projected)
