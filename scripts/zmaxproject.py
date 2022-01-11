# Z-max projection

from pathlib import Path
from tqdm import tqdm

from centrack.data import DataSet, Field, Condition, PixelSize
from centrack.utils import is_tif, extract_filename


def main():
    path_dataset = Path('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1')
    markers = 'DAPI+CEP152+GTU88+PCNT'.split('+')
    conditions = Condition(markers=markers,
                           genotype='RPE1wt',
                           pixel_size=PixelSize(.1025, 'um'))
    dataset = DataSet(path_dataset, condition=conditions)
    dataset.projections.mkdir(exist_ok=True)

    files = [file for file in dataset.projections.iterdir() if is_tif(file)]

    pbar = tqdm(files)

    for file in pbar:
        field = Field(file, dataset=dataset)
        data = field.data
        projected = data.max(axis=2)

        file_name = extract_filename(file)
    #
    #     dest_name = f'{file_name}_max.ome.tif'
    #     path_projected = dataset.projections / dest_name
    #     tf.imwrite(path_projected, projected, photometric='minisblack')


if __name__ == '__main__':
    main()
