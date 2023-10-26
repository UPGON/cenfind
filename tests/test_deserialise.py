import json
import numpy as np
import cv2


def test_load_contours():
    with open('data/cilia/predictions/nuclei/RPE1-GFPCep135_NC_48h_100k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_1_MMStack_10-Pos_000_001_max_C0.json', 'r') as f:
        contours = json.load(f)
    assert type(contours) == dict
    cnts = [np.array(cnt['contour']) for cnt in contours.values()]
    bg = np.zeros((2048, 2048), dtype='uint8')
    cv2.drawContours(bg, cnts, -1, 255, -1)
    cv2.imwrite('../out/test.png', bg)
