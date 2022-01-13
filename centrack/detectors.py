from abc import ABC, abstractmethod

from cv2 import cv2
import scipy.ndimage
from skimage import img_as_float
from skimage.feature import peak_local_max

from centrack.annotation import Centre, Contour
from centrack.utils import contrast


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
        transformed = self.plane
        transformed = cv2.GaussianBlur(transformed, (31, 31), 0)
        transformed = contrast(transformed)
        th, mask = cv2.threshold(transformed, 200, 255, 0)
        return mask

    def detect(self):
        transformed = self._mask()
        contours, hierarchy = cv2.findContours(transformed,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        return [Contour(c, self.organelle, c_id, confidence=-1) for c_id, c in enumerate(contours)]
