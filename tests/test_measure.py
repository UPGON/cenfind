import numpy as np

from cenfind.core.measure import Assigner
from cenfind.core.structures import Centriole, Nucleus


def _square_contour(x0, y0, size):
    return np.array(
        [[[x0, y0]], [[x0, y0 + size]], [[x0 + size, y0 + size]], [[x0 + size, y0]]],
        dtype=np.int32,
    )


def test_assign_centrioles_to_nearest_nucleus(make_field):
    field = make_field(shape=(2, 200, 200))

    nucleus_a = Nucleus(field=field, channel=0, contour=_square_contour(10, 10, 40), index=0, label="Nucleus")
    nucleus_b = Nucleus(field=field, channel=0, contour=_square_contour(100, 100, 40), index=1, label="Nucleus")

    # Both inside nucleus_a's square (rows/cols 10-50).
    centriole_a1 = Centriole(field=field, channel=1, centre=(15, 15), index=0, label="Centriole")
    centriole_a2 = Centriole(field=field, channel=1, centre=(45, 20), index=1, label="Centriole")
    # Inside nucleus_b's square (rows/cols 100-140).
    centriole_b1 = Centriole(field=field, channel=1, centre=(120, 120), index=2, label="Centriole")
    # Far from both nuclei.
    centriole_far = Centriole(field=field, channel=1, centre=(5, 190), index=3, label="Centriole")

    centrioles = [centriole_a1, centriole_a2, centriole_b1, centriole_far]
    nuclei = [nucleus_a, nucleus_b]

    assigner = Assigner(centrioles=centrioles, nuclei=nuclei, vicinity=0)
    assignment = assigner.assign_centrioles()

    assigned = dict(assignment)
    assert assigned[centriole_a1.index] == nucleus_a.index
    assert assigned[centriole_a2.index] == nucleus_a.index
    assert assigned[centriole_b1.index] == nucleus_b.index
    assert assigned[centriole_far.index] == -1


def test_score_nuclei_counts_assigned_centrioles(make_field):
    field = make_field(shape=(2, 200, 200))

    nucleus_a = Nucleus(field=field, channel=0, contour=_square_contour(10, 10, 40), index=0, label="Nucleus")
    nucleus_b = Nucleus(field=field, channel=0, contour=_square_contour(100, 100, 40), index=1, label="Nucleus")

    centriole_a1 = Centriole(field=field, channel=1, centre=(15, 15), index=0, label="Centriole")
    centriole_a2 = Centriole(field=field, channel=1, centre=(45, 20), index=1, label="Centriole")

    assigner = Assigner(centrioles=[centriole_a1, centriole_a2], nuclei=[nucleus_a, nucleus_b], vicinity=0)
    scores = assigner.score_nuclei("field_x", channel=1)

    scores_by_nucleus = dict(zip(scores["nucleus"], scores["score"]))
    assert scores_by_nucleus[nucleus_a.index] == 2
    assert scores_by_nucleus[nucleus_b.index] == 0
    assert (scores.index.get_level_values("field") == "field_x").all()
    assert (scores.index.get_level_values("channel") == 1).all()


def test_assign_centrioles_empty_inputs(make_field):
    field = make_field(shape=(2, 200, 200))
    nucleus_a = Nucleus(field=field, channel=0, contour=_square_contour(10, 10, 40), index=0, label="Nucleus")

    assigner = Assigner(centrioles=[], nuclei=[nucleus_a], vicinity=0)
    assert assigner.assign_centrioles() == []
