import numpy as np
import sys
import tifffile as tf
from aicsimageio import AICSImage

from skimage import io
from skimage import morphology
from skimage.filters import gaussian
from skimage import threshold_adaptive
from skimage.feature import peak_local_max
from skimage.morphology import watershed, remove_small_objects
import skimage.segmentation
from scipy import ndimage

# Functional parameters
intensity_blur_rad = 0.
radius_threshold = 75.
threshold = -10
watershed_noise_tolerance = 3
size_min = 25


def detect_foci(data):
    foci_list = []
    return foci_list


def detect_nuclei(data):
    # Gaussian filter
    if intensity_blur_rad >= 1:
        data = 255 * gaussian(data, sigma=intensity_blur_rad)

    # Adaptive threshold
    mask = threshold_adaptive(data, radius_threshold, offset=Thr).astype(np.uint8)

    # Binary watershed
    distance = ndimage.distance_transform_edt(mask)
    distance = gaussian(distance, sigma=watershed_noise_tolerance)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
    markers = morphology.label(local_maxi)
    nuclei_labels = watershed(-distance, markers, mask=mask)
    nuclei_labels = nuclei_labels.astype(np.uint16)
    nuclei_labels = remove_small_objects(nuclei_labels, min_size=size_min)
    nuclei_labels = skimage.segmentation.relabel_sequential(nuclei_labels)[0]
    nuclei_labels = nuclei_labels.astype(np.uint16)

    return nuclei_labels


def main():
    channel_id_nuclei = 0
    channel_id_foci = 1

    field = AICSImage('/Volumes/work/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1/projections_channel/DAPI/tif/RPE1wt_CEP63+CETN2+PCNT_1_000_000_max_C0.tif')
    data_nuclei = field.get_image_data('ZYX', C=channel_id_nuclei).max(axis=0).squeeze()
    data_foci = field.get_image_data('ZYX', C=channel_id_foci).max(axis=0).squeeze()

    nuclei = detect_nuclei(data_nuclei)
    foci = detect_foci(data_foci)

    return 0


if __name__ == '__main__':
    sys.exit(main())
