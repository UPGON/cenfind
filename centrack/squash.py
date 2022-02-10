import argparse
import logging
import sys
from pathlib import Path
from time import sleep

import tifffile as tf
from tqdm import tqdm

from centrack.status import build_name, fetch_files

logging.basicConfig(format='%(levelname)s: %(message)s')

logger_cli = logging.getLogger(__name__)
logger_cli.setLevel(logging.INFO)

logger_tifffile = logging.getLogger("tifffile")
logger_tifffile.setLevel(logging.ERROR)


def project(path):
    """
    Reads an OME.tiff and applies a max projection for the z-axis.
    :returns the pixel size in centimeters and the CYX-array.
    """
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


def write_projection(dst, data, pixel_size=None):
    """
    Writes the projection to the disk.
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
    parser.add_argument('--top', '-t',
                        action='store_true',
                        help='Write the projections in one single folder `projections` immediately below the source.',
                        )
    parser.add_argument('--recursive', '-r',
                        action='store_true',
                        default=False,
                        help='Process all ome.tif files under the directory.',
                        )
    parser.add_argument('--force', '-f',
                        action='store_true',
                        )
    parser.add_argument('--mock', '-m',
                        action='store_true',
                        help='Emulate the iteration with no heavy I/O nor max projection.'
                        )

    return parser.parse_args()


def cli():
    args = parse_args()
    if args.mock:
        logger_cli.warning('Mock mode: Files are not processed.')

    path_raw = args.source / 'raw'
    if not path_raw.exists():
        sys.exit(
            'Please move the raw ome.tif files or tree thereof to `raw/`')

    files = fetch_files(path_raw, recursive=args.recursive)

    if args.top:
        projections_path = args.source / 'projections'
        projections_path.mkdir(exist_ok=True)
        logger_cli.info('Projections will be saved in %s', projections_path)
    else:
        projections_path = None

    if files:
        logger_cli.info('Found %s files', len(files))
    else:
        logger_cli.warning('No file found...')
        logger_cli.info('The recursive flag is currently %s', args.recursive)

    pbar = tqdm(files)

    for path in pbar:
        # logger_cli.info('Processing: %s', path.name)
        dst_name = build_name(path)

        if not args.top:
            projections_path = path

        path_dst = projections_path / dst_name

        if not args.force and path_dst.exists():
            logger_cli.warning('Projection already exists (%s)', path_dst)
            should_overwrite = input(
                'Do you want to overwrite the existing projection? (y/n): ')
            if should_overwrite in ('n', 'no'):
                logging.warning('Skipping %s', path)
                continue

        if args.mock:
            sleep(.1)
            continue

        pixel_size_cm, projection = project(path)
        write_projection(path_dst, projection, pixel_size_cm)


if __name__ == '__main__':
    cli()
