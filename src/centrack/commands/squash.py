import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from tqdm import tqdm

from centrack.utils.status import fetch_files

logging.basicConfig(format='%(levelname)s: %(message)s')

logger_cli = logging.getLogger(__name__)
logger_cli.setLevel(logging.INFO)

logger_tifffile = logging.getLogger("tifffile")
logger_tifffile.setLevel(logging.ERROR)


def extract_pixel_size(path):
    """Extract the pixel size of an ome.tif file"""
    with tf.TiffFile(path) as file:
        try:
            micromanager_metadata = file.micromanager_metadata
            pixel_size_um = micromanager_metadata['Summary']['PixelSize_um']
        except KeyError:
            pixel_size_um = None
            logging.warning('No pixel size could be found')

    if pixel_size_um:
        pixel_size_cm = pixel_size_um / 1e4
    else:
        pixel_size_cm = None

    return pixel_size_cm


def reshape(path: Path) -> np.ndarray:
    """
    Determine the shape of the stack and reshape it as CZYX.
    """
    with tf.TiffFile(path) as file:
        shape = file.series[0].shape
        order = file.series[0].axes
        data = file.asarray()

    dimensions_found = set(order.lower())
    dimensions_expected = set('czyx')

    if dimensions_found != dimensions_expected:
        if dimensions_found != set('zyx'):
            raise ValueError(
                f"Dimension mismatch in {path}: found: {dimensions_found} vs expected: {dimensions_expected}")

    if order == 'ZCYX':
        z, c, y, x = shape
    elif order == 'CZYX':
        c, z, y, x = shape
    elif order == 'ZYX':
        c = 1
        z, y, x = shape
        data = np.expand_dims(data, 0)
    else:
        raise ValueError(f'Order is not understood {order}')

    reshaped = data.reshape((c, z, y, x))

    return reshaped


def project(data: np.ndarray, p) -> np.ndarray:
    """Max project a CZYX numpy stack."""
    _data = data.copy()
    return _data.max(axis=1)


def process(path, projection_type):
    pixel_size_cm = None
    data_reshaped = reshape(path)
    if projection_type == 'max':
        projection = np.max(data_reshaped, axis=1)
        return projection, pixel_size_cm
    if projection_type == 'sum':
        projection = np.sum(data_reshaped, axis=1)
        return projection, pixel_size_cm


def write_projection(dst: Path, data: np.ndarray, pixel_size=None) -> None:
    """
    Writes the projection to the disk.
    :param dst: the path of the output file
    :param data: the data to write to dst
    :param pixel_size: the pixel size in cm

    :return None
    """

    if pixel_size:
        res = (1 / pixel_size, 1 / pixel_size, 'CENTIMETER')
    else:
        res = None

    tf.imwrite(dst, data, photometric='minisblack', resolution=res)


def cli():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tiff files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source',
                        type=Path,
                        help='Path to the dataset folder; the parent of `raw`.',
                        )
    parser.add_argument('format',
                        type=str,
                        help='Format of the raw files, e.g., `.ome.tif` or `.stk`')
    parser.add_argument('projection_type',
                        type=str,
                        help='Type of projection: `max`, `sum`, `mean`')

    args = parser.parse_args()

    path_raw = args.source / 'raw'
    if not path_raw.exists():
        raise FileNotFoundError(
            f'raw/ folder not found, please make sure to move the ome.tif files in raw/.')

    path_projections = args.source / 'projections'
    if not path_projections.exists():
        logger_cli.info('Create projections folder')
        path_projections.mkdir()

    files = fetch_files(path_raw, file_type=args.format)

    for path in tqdm(files):
        projection, pixel_size_cm = process(path, args.projection_type)
        dst_name = path.name.replace(args.format, f"_{args.projection_type}.tif")
        write_projection(path_projections / dst_name, projection, pixel_size_cm)


if __name__ == '__main__':
    cli()
