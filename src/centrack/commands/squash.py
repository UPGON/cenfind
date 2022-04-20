import numpy as np
import tifffile as tf
from pathlib import Path


def correct_axes(data: np.ndarray):
    z, c, y, x = data.shape
    corrected = data.copy()
    return corrected.reshape((c, z, y, x))


def extract_pixel_size(path: Path):
    with tf.TiffFile(path) as f:
        mm_metadata_summary = f.micromanager_metadata['Summary']
    try:
        pixel_size_um = mm_metadata_summary['PixelSize_um']
    except KeyError:
        pixel_size_um = None

    if pixel_size_um:
        return pixel_size_um / 1e4
    else:
        return None


def extract_axes_order(path):
    with tf.TiffFile(path) as f:
        axes_order = f.series[0].axes
    return axes_order


def load_ome_tif(path: Path):
    """
    Read an OME tif file from the disk into a numpy array
    :param path:
    :return:
    """
    with tf.TiffFile(path) as f:
        data = f.asarray()
    return data


def squash(data: np.ndarray):
    """
    Squash a numpy array along the z-axis using max function.
    :param data:
    :return:
    """
    return data.max(axis=1)


def read_ome_tif(path_ome):
    data = load_ome_tif(path_ome)
    pixel_size = extract_pixel_size(path_ome)
    axes_order = extract_axes_order(path_ome)
    if axes_order == 'ZCYX':
        data = correct_axes(data)
    return pixel_size, data

def collect_ome_tif(path_dataset: Path):
    """
    Collect all OME.tif files located under raw/
    :param path_dataset:
    :return:
    """
    path_raw = path_dataset / 'raw'

    if not path_raw.exists():
        raise FileNotFoundError(path_raw.resolve())

    return (f for f in path_dataset.rglob('*.ome.tif')
                 if not f.name.startswith('.'))