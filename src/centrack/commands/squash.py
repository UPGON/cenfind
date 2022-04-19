import numpy as np
import tifffile as tf
from pathlib import Path

def correct_axes(data: np.ndarray):
    z, c, y, x = data.shape
    corrected = data.copy()
    return corrected.reshape((c, z, y, x))

def read_ome_tif(path: Path):
    """
    Read an OME tif file from the disk into a numpy array
    :param path:
    :return:
    """
    with tf.TiffFile(path) as f:
        mm_metadata_summary = f.micromanager_metadata['Summary']
        pixel_size = mm_metadata_summary['PixelSize_um']
        order = f.series[0].axes
        data = f.asarray()
    return pixel_size, order, data

def squash(data: np.ndarray):
    """
    Squash a numpy array along the z-axis using max function.
    :param data:
    :return:
    """
    return data.max(axis=1)
