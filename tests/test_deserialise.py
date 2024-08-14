import json
import numpy as np
import cv2
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_load_contours():
    with open(ROOT_DIR / 'data/cilia/predictions/nuclei/RPE1-GFPCep135_NC_48h_100k_mCETN2-AF488_rArl13b-AF568_gCEP164-AF647_1_MMStack_10-Pos_000_001_max_C0.json', 'r') as f:
        contours = json.load(f)
    assert type(contours) == dict
    cnts = [np.array(cnt['contour']) for cnt in contours['nuclei'].values()]
    bg = np.zeros((2048, 2048), dtype='uint8')
    cv2.drawContours(bg, cnts, -1, 255, -1)
    cv2.imwrite('../out/test.png', bg)
