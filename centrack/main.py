from pathlib import Path
from centrack.data import DataSet, Field, Plane, Condition, PixelSize


def cli():
    markers = 'DAPI+CEP152+GTU88+PCNT'.split('+')
    conditions = Condition(markers=markers,
                           genotype='RPE1wt',
                           pixel_size=PixelSize(.1025, 'um'))

    dataset_path = Path('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1')
    field_name = 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000_max.tif'
    ds = DataSet(dataset_path, condition=conditions)
    field = Field(dataset_path / field_name, dataset=ds)
    foci = Plane(field, 'CEP152')
    data = foci.data


    foci_mask = (data
                 .blur_median(3)
                 .maximum_filter(size=5)
                 .contrast()
                 .threshold(threshold=20)
                 )


if __name__ == '__main__':
    cli()
