import numpy
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
    data = tf.imread(path)
    return data

def squash(data: np.ndarray):
    """
    Squash a numpy array along the z-axis using max function.
    :param data:
    :return:
    """
    return data.max(axis=1)
