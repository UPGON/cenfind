from cenfind.core.data import Dataset


def test_label():
    dataset = Dataset('/data1/centrioles/IF_2022_10_17_U2OS_WTsiSTIL_sync2h_DNA_rPlk4AF488_mCentrn2AF568_2022_10_25_1')
    field = dataset.fields[0]
    label = field.score()
    assert 2 == 2
