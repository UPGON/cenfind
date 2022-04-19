import numpy
import tifffile as tf
from pathlib import Path

def read_ome_tif(path: Path):
    """
    Read an OME tif file from the disk into a numpy array
    :param path:
    :return:
    """
    data = tf.imread(path)
    return data

def squash(data: numpy.ndarray):
    """
    Squash a numpy array along the z-axis using max function.
    :param data:
    :return:
    """
    return data.max(axis=1)
