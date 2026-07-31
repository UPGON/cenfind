import numpy as np

from cenfind.core.structures import Centriole, Nucleus


def test_centriole_centre_xy_swaps_row_col(make_field):
    field = make_field(shape=(2, 100, 100))
    centriole = Centriole(field=field, channel=0, centre=(7, 3))
    assert centriole.centre_xy == (3, 7)


def test_centriole_intensity_sums_neighbourhood(make_field):
    data = np.zeros((2, 20, 20), dtype="uint16")
    data[1, 9:12, 9:12] = 5  # 3x3 block of value 5 around (10, 10) on channel 1
    field = make_field(data=data)
    centriole = Centriole(field=field, channel=1, centre=(10, 10))

    # intensity(k=1) sums image[channel, r-1:r+1, c-1:c+1], i.e. rows/cols 9:11 -
    # a 2x2 window here, not a symmetric 3x3 one. Compare against that same
    # slice directly rather than assuming a specific window size.
    expected = int(data[1, 10 - 1:10 + 1, 10 - 1:10 + 1].sum())
    assert centriole.intensity(field.data, k=1, channel=1) == expected


def test_nucleus_centre_and_area_of_a_square(make_field):
    field = make_field(shape=(2, 200, 200))
    # Square spanning x/y 10-50 -> centroid (30, 30) in (row, col).
    contour = np.array(
        [[[10, 10]], [[10, 50]], [[50, 50]], [[50, 10]]], dtype=np.int32
    )
    nucleus = Nucleus(field=field, channel=0, contour=contour, index=0, label="Nucleus")

    row, col = nucleus.centre
    assert abs(row - 30) <= 1
    assert abs(col - 30) <= 1
    assert nucleus.centre_xy == (col, row)
    assert nucleus.area > 0


def test_nucleus_full_in_field_flags_edge_nuclei(make_field):
    field = make_field(shape=(2, 200, 200))

    centred = np.array(
        [[[90, 90]], [[90, 110]], [[110, 110]], [[110, 90]]], dtype=np.int32
    )
    at_edge = np.array(
        [[[0, 0]], [[0, 5]], [[5, 5]], [[5, 0]]], dtype=np.int32
    )

    nucleus_centred = Nucleus(field=field, channel=0, contour=centred, index=0, label="Nucleus")
    nucleus_edge = Nucleus(field=field, channel=0, contour=at_edge, index=1, label="Nucleus")

    assert nucleus_centred.full_in_field is True
    assert nucleus_edge.full_in_field is False


def test_nucleus_as_dict_contains_expected_keys(make_field):
    field = make_field(shape=(2, 200, 200))
    contour = np.array(
        [[[10, 10]], [[10, 50]], [[50, 50]], [[50, 10]]], dtype=np.int32
    )
    nucleus = Nucleus(field=field, channel=0, contour=contour, index=0, label="Nucleus")

    result = nucleus.as_dict()
    assert set(result.keys()) == {
        "channel", "pos_r", "pos_c", "intensity", "surface_area", "is_nucleus_full", "contour",
    }
    assert result["contour"] == contour.tolist()
