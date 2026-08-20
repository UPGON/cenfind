import cv2
import numpy as np

from cenfind.core.detectors import extract_cilia


def test_extract_cilia_detects_elongated_blob(make_field):
    data = np.zeros((2, 300, 300), dtype="uint16")
    cv2.ellipse(data[1], (150, 150), (60, 10), 30, 0, 360, 4000, -1)
    field = make_field(data=data)

    cilia = extract_cilia(field, channel=1)

    assert len(cilia) == 1
    row, col = cilia[0].centre
    assert abs(row - 150) < 5
    assert abs(col - 150) < 5
    assert cilia[0].label == "Cilium"


def test_extract_cilia_blank_image_detects_nothing(make_field):
    data = np.zeros((2, 300, 300), dtype="uint16")
    field = make_field(data=data)

    assert extract_cilia(field, channel=1) == []
