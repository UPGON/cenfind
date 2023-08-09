import json
import numpy as np
import cv2


def test_load_contours():
    with open('/Users/buergy/Dropbox/epfl/projects/cenfind/data/dataset_test/predictions/nuclei/RPE1wt_CEP63+CETN2+PCNT_1_000_000_C0.json', 'r') as f:
        contours = json.load(f)
    assert type(contours) == dict
    cnts = [np.array(cnt) for cnt in contours.values()]
    bg = np.zeros((2048, 2048), dtype='uint8')
    cv2.drawContours(bg, cnts, -1, 255, -1)
    cv2.imwrite('../out/test.png', bg)
