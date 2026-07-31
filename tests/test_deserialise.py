import json

import cv2
import numpy as np

from cenfind.core.serialise import save_contours
from cenfind.core.structures import Nucleus


def test_save_and_load_contours(tmp_path, make_field):
    field = make_field(shape=(2, 64, 64))
    contour = np.array([[[10, 10]], [[10, 30]], [[30, 30]], [[30, 10]]], dtype=np.int32)
    nucleus = Nucleus(field=field, channel=0, contour=contour, index=0, label="Nucleus")

    dst = tmp_path / "contours.json"
    save_contours(dst, [nucleus])

    with open(dst, "r") as f:
        contours = json.load(f)

    assert isinstance(contours, dict)
    assert set(contours["nuclei"].keys()) == {"0"}

    reloaded_contour = np.array(contours["nuclei"]["0"]["contour"])
    np.testing.assert_array_equal(reloaded_contour, contour)

    # The reloaded contour should draw the same filled area as the original.
    original_mask = np.zeros((64, 64), dtype="uint8")
    cv2.drawContours(original_mask, [contour], -1, 255, -1)

    reloaded_mask = np.zeros((64, 64), dtype="uint8")
    cv2.drawContours(reloaded_mask, [reloaded_contour], -1, 255, -1)

    np.testing.assert_array_equal(original_mask, reloaded_mask)
