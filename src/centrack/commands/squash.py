import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from tqdm import tqdm

from centrack.commands.status import build_name, fetch_files

logging.basicConfig(format='%(levelname)s: %(message)s')

logger_cli = logging.getLogger(__name__)
logger_cli.setLevel(logging.INFO)

logger_tifffile = logging.getLogger("tifffile")
logger_tifffile.setLevel(logging.ERROR)


def project(path: Path) -> (float, np.ndarray):
    """
    Reads an OME.tiff and applies a max projection for the z-axis.
    :param path: Path object of an OME.tif file
    :returns the pixel size in centimeters and the CYX-array.
    """
    if not path.exists:
        raise FileNotFoundError(path)

    with tf.TiffFile(path) as file:
        shape = file.series[0].shape
        order = file.series[0].axes
        dimensions_found = set(order.lower())
        dimensions_expected = set('czyx')

        if dimensions_found != dimensions_expected:
            raise ValueError(
                f"Dimension mismatch: found: {dimensions_found} vs expected: {dimensions_expected}")

        if order == 'ZCYX':
            z, c, y, x = shape
        elif order == 'CZYX':
            c, z, y, x = shape
        else:
            raise ValueError(f'Order is not understood {order}')

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

        data = file.asarray()

        logging.info('Order: %s Shape: %s', order, shape)
        _data = data.reshape((c, z, y, x))

    projection = _data.max(axis=1)

    return pixel_size_cm, projection


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


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     description='Project OME.tiff files',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('source',
                        type=Path,
                        help='Path to the dataset folder; the parent of `raw`.',
                        )

    return parser.parse_args()


def cli():
    args = parse_args()

    path_raw = args.source / 'raw'

    if not path_raw.exists():
        raise FileNotFoundError(
            f'raw/ folder not found, please make sure to move the ome.tif files in raw/.')

    path_projections = args.source / 'projections'
    if not path_projections.exists():
        logger_cli.info('Create projections folder')
        path_projections.mkdir()

    files = fetch_files(path_raw, file_type='ome.tif')

    pbar = tqdm(files)

    for path in pbar:
        pixel_size_cm, projection = project(path)

        dst_name = build_name(path)
        path_dst = path_projections / dst_name
        write_projection(path_dst, projection, pixel_size_cm)


if __name__ == '__main__':
    cli()
