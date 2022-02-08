from outline import Centre
from score import assign


def test_assign():
    nuclei_list = [(1, 10), (4, 5), (3, 4), (10, 2)]
    foci_list = [(2, 7), (3, 7), (20, 5), (30, 1), (2, 14), (2, 6)]

    nuclei_list = [Centre(t, idx=i, label=f'nucleus {i}', confidence=-1) for
                   i, t in enumerate(nuclei_list)]
    foci_list = [Centre(t, idx=i, label=f'centriole {i}', confidence=-1) for
                 i, t in enumerate(foci_list)]

    result = assign(foci_list, nuclei_list)
    print(result)
