from abc import ABC, abstractmethod

import cv2
import numpy as np
from skimage import img_as_float
from skimage.feature import peak_local_max

from centrack.annotation import Centre


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
        transformed = (self.plane
                       .blur_median(3)
                       .maximum_filter(size=5)
                       .contrast()
                       .threshold(threshold=50)
                       )
        return transformed

    def detect(self, interpeak_min=3):
        image = self.plane.data
        mask = self._mask().data
        masked = cv2.bitwise_and(image, image, mask=mask)

        centrioles_float = img_as_float(masked)
        foci_coords = np.fliplr(peak_local_max(centrioles_float, min_distance=interpeak_min))

        return [Centre(f, f_id, self.organelle, confidence=-1) for f_id, f in enumerate(foci_coords)]
