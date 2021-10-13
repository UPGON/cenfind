# Z-max projection

import numpy as np
from tqdm import tqdm
from aicsimageio import AICSImage

from centrack.data import DataSet, Field


if __name__ == '__main__':

    dataset = DataSet('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1')
    dataset.projections.mkdir(exist_ok=True)

    files = dataset.fields

    pbar = tqdm(files)

    for file in pbar:
        field = AICSImage(file)
        projected = field.data.max(axis=2)
        projected = np.expand_dims(projected, 2)
        file_name_core = file.name.removesuffix(''.join(file.suffixes))

        dest_name = f'{file_name_core}_max.ome.tif'
        path_projected = dataset.projections / dest_name
        AICSImage(projected).save(path_projected)
