from pathlib import Path

from centrack import file_read, markers_from


path_root = Path('/Volumes/work/datasets')
dataset_name = 'RPE1p53_Cnone_CEP63+CETN2+PCNT_1'
path_raw = path_root / dataset_name / 'raw'

path_main = path_raw / 'RPE1p53+Cnone_CEP63+CETN2+PCNT_1_MMStack_6-Pos_000_000.ome.tif'
path_companion = path_raw / 'RPE1p53+Cnone_CEP63+CETN2+PCNT_1_MMStack_6-Pos_000_001.ome.tif'


def test_markers_from():
    theoretical = {0: 'DAPI',
                   1: 'CEP63',
                   2: 'CETN2',
                   3: 'PCNT'}

    actual = markers_from(dataset_name)

    assert theoretical == actual


def test_file_read():
    shape = (4, 67, 2048, 2048)
    data_main = file_read(path_main)
    data_companion = file_read(path_companion)

    assert data_main.shape == shape
    assert data_companion.shape == shape

    assert data_main.shape == data_companion.shape, 'Shape not identical'