from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage
from csbdeep.utils import normalize
from cv2 import cv2
from skimage import img_as_float
from skimage.feature import peak_local_max
from stardist.models import StarDist2D

from centrack.annotation import Centre, Contour
from centrack.utils import contrast


def mat2gray(image):
    """Normalize to the unit interval and return a float image"""
    return cv2.normalize(image, None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


class Detector(ABC):
    def __init__(self, plane, organelle):
        self.plane = plane
        self.organelle = organelle

    @abstractmethod
    def _mask(self):
        pass

    @abstractmethod
    def detect(self):
        pass


class FocusDetector(Detector):
    """Combine a preprocessing and a detection step and return a list of centres."""

    def _mask(self):
        transformed = self.plane
        transformed = cv2.GaussianBlur(transformed, (3, 3), 0)
        transformed = scipy.ndimage.maximum_filter(transformed, size=5)
        transformed = contrast(transformed)
        th, transformed = cv2.threshold(transformed, 50, 255, cv2.THRESH_BINARY)

        return transformed

    def detect(self, interpeak_min=3):
        image = self.plane
        mask = self._mask()
        masked = cv2.bitwise_and(image, image, mask=mask)

        centrioles_float = img_as_float(masked)
        foci_coords = peak_local_max(centrioles_float, min_distance=interpeak_min)

        return [Centre(f, f_id, self.organelle, confidence=-1) for f_id, f in enumerate(foci_coords)]


class NucleiDetector(Detector):
    """
    Threshold a DAPI image and run contour detection.
    """

    def _mask(self):
        image = self.plane
        image32f = image.astype(float)
        mu = cv2.GaussianBlur(image32f, (101, 101), 0)
        mu2 = cv2.GaussianBlur(mu * mu, (31, 31), 0)
        sigma = cv2.sqrt(mu2 - mu * mu)
        cv2.imwrite('./out/sigma.png', sigma)

        th, mask = cv2.threshold(sigma, 200, 255, 0)
        cv2.imwrite('./out/mask.png', mask)

        return mask

    def detect(self):
        transformed = self._mask()
        contours, hierarchy = cv2.findContours(transformed,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        return [Contour(c, self.organelle, c_id, confidence=-1) for c_id, c in enumerate(contours)]


class NucleiStardistDetector(Detector):
    """
    Resize a DAPI image and run StarDist
    """

    def _mask(self):
        return cv2.resize(self.plane, dsize=(256, 256),
                          fx=1, fy=1,
                          interpolation=cv2.INTER_NEAREST)

    def detect(self):
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        transformed = self._mask()

        transformed = transformed
        labels, coords = model.predict_instances(normalize(transformed))

        nuclei_detected = cv2.resize(labels, dsize=(2048, 2048),
                                     fx=1, fy=1,
                                     interpolation=cv2.INTER_NEAREST)

        labels_id = np.unique(nuclei_detected)
        cnts = []
        for nucleus_id in labels_id:
            if nucleus_id == 0:
                continue
            submask = np.zeros_like(nuclei_detected, dtype='uint8')
            submask[nuclei_detected == nucleus_id] = 255
            contour, hierarchy = cv2.findContours(submask,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            cnts.append(contour[0])
        contours = tuple(cnts)
        return [Contour(c, self.organelle, c_id, confidence=-1) for c_id, c in enumerate(contours)]
