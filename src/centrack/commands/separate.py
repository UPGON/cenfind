import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from cv2 import cv2
from skimage import exposure
from tqdm import tqdm

from centrack.commands.status import DataSet, fetch_files
from centrack.commands.outline import contrast

logging.basicConfig(level=logging.DEBUG)


def improve(image, percentile=(.1, 99.9)):
    percentiles = np.percentile(image, percentile)
    return exposure.rescale_intensity(image, in_range=tuple(percentiles), out_range='uint8')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path, help='Path to the max projection')
    parser.add_argument('markers', type=str, nargs='*', help='Markers')
    return parser.parse_args()


def cli(args=None):
    args = parse_args()
    dataset = DataSet(args.source)
    markers = args.markers

    files = fetch_files(dataset.projections, file_type='_max.tif')
    pbar = tqdm(files)

    for field in pbar:
        name_core = field.name.rstrip('.tif')
        # data = tf.imread(field)
        #
        for c, marker in enumerate(markers):
            # channel = marker.protein
            # print(channel)
            path_projections_channel = path_dataset / 'projections_channel' / marker
        #
            path_projections_channel.mkdir(exist_ok=True, parents=True)
            (path_projections_channel / 'tif').mkdir(exist_ok=True)
            (path_projections_channel / 'png').mkdir(exist_ok=True)
        #
            file_name_dst = f"{name_core}_C{c}"
        #
            data_yxc = np.moveaxis(data, 0, -1)
            plane = data_yxc[:, :, c]
            tf.imwrite(path_projections_channel / 'tif' / (file_name_dst + ".tif"), plane)
        #
        #     res = contrast(plane)
        #     cv2.putText(res, f"{file_name_dst} ({channel})", (200, 200), cv2.QT_FONT_NORMAL, .8,
        #                 color=(255, 255, 255),
        #                 thickness=2)
        #
        #     cv2.imwrite(str(path_projections_channel / 'png' / (file_name_dst + ".png")), res)
        #     logging.info(f'Saved: {name_core} channel: {c}')


if __name__ == '__main__':
    cli()
