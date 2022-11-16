from cenfind.core.data import Dataset
from cenfind.core.measure import extract_nuclei
from types import SimpleNamespace
import numpy as np
from skimage.draw import disk
from matplotlib import pyplot as plt
from cenfind.core.measure import assign

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# from spotipy.utils import points_to_label
from cenfind.core.outline import Contour, Centre
import cv2


def points_to_label(points, shape=None, max_distance=3):
    points = np.asarray(points).astype(np.int32)
    assert points.ndim == 2 and points.shape[1] == 2

    if shape is None:
        mi = np.min(points, axis=0)
        ma = np.max(points, axis=0)
        points = points - mi
        shape = (ma - mi).astype(int)

    im = np.zeros(shape, np.uint16)

    for i, p in enumerate(points):
        rr, cc = disk(p, max_distance, shape=shape)
        im[rr, cc] = i + 1
    return im


def points_matching_v2(p1, p2, cutoff_distance=5):
    """ finds matching that minimizes sum of mean squared distances"""

    D = cdist(p1, p2, metric='sqeuclidean')

    if D.size > 0:
        D[D > cutoff_distance ** 2] = 1e10 * (1 + D.max())

    i, j = linear_sum_assignment(D)
    valid = D[i, j] <= cutoff_distance ** 2
    i, j = i[valid], j[valid]

    res = SimpleNamespace()

    tp = len(i)
    fp = len(p2) - tp
    fn = len(p1) - tp
    res.tp = tp
    res.fp = fp
    res.fn = fn
    res.accuracy = tp / (tp + fp + fn) if tp > 0 else 0
    res.precision = tp / (tp + fp) if tp > 0 else 0
    res.recall = tp / (tp + fn) if tp > 0 else 0
    res.f1 = (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0

    res.dist = np.sqrt(D[i, j])
    res.mean_dist = np.mean(res.dist) if len(res.dist) > 0 else 0

    res.false_negatives = tuple(set(range(len(p1))).difference(set(i)))
    res.false_positives = tuple(set(range(len(p2))).difference(set(j)))
    res.matched_pairs = tuple(zip(i, j))
    return res


def test_assign():
    ds = Dataset('/data1/centrioles/RPE1wt_CEP63+CETN2+PCNT_1/')
    field = ds.fields[0]
    factor = 256
    nuclei_channel = 0
    mask = field.mask(0)
    centres, nuclei = extract_nuclei(field, nuclei_channel, factor, annotation=mask)

    foci = field.annotation(1)
    foci_label = points_to_label(foci, shape=mask.shape, max_distance=6)
    foci_mask = (255 * (foci_label > 0)).astype('uint8')
    contours, hierarchy = cv2.findContours(foci_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    contours = tuple(contours)
    centrosomes = [Contour(c, 'Centrosome', c_id, confidence=-1) for c_id, c in
                enumerate(contours)]

    centrosomes_centres = [c.centre.to_numpy() for c in centrosomes]
    centrosomes_centres = np.stack(centrosomes_centres)
    foci = [Centre((r, c), f_id, 'Centriole') for f_id, (r, c) in enumerate(foci)]
    assigned = assign(foci, nuclei, -50, .1025)

    centrosomes_label = points_to_label(centrosomes_centres, shape=mask.shape, max_distance=6)

    res = points_matching_v2(foci, foci + np.random.normal(0, 5))

    print(centres)
