import numpy as np

from cenfind.core.measure import save_foci
from cenfind.core.outline import Point


def test_save_foci():
    foci_list = [Point((x, y)) for x, y in np.random.randint(1, 10, size=(10, 2))]
    assert len(foci_list) == 10
    save_foci(foci_list, './saved_foci.tsv')

    foci_list_empty = []
    assert len(foci_list_empty) == 0
    save_foci(foci_list_empty, './foci_list_empty.tsv')
