import argparse
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import tifffile as tf
import numpy as np

from centrack import sharp_planes
from utils import fov_read, channels_combine


def args_parse():
    parser = argparse.ArgumentParser(description='Image processing pipeline')

    parser.add_argument('-d', '--name', required=True,
                        help='Name of the dataset')
    parser.add_argument('-p', '--path', required=True,
                        help='Path to the dataset location')
    parser.add_argument('-1', '--first', action='store_true',
                        help='Flag to process only the first field of view')
    parser.add_argument('-c', '--color', action='store_true',
                        help='Flag to write RGB images with the centriole markers')
    return vars(parser.parse_args())


def main(args):
    dataset_name = args['name']

    path_root = Path(args['path'])
    path_raw = path_root / dataset_name / 'raw'

    path_projections = path_root / dataset_name / 'projected'
    path_projections.mkdir(exist_ok=True)

    path_rgb = path_root / dataset_name / 'color'
    path_rgb.mkdir(exist_ok=True)

    files = sorted(tuple(file for file in path_raw.iterdir()
                         if file.name.endswith('.tif')
                         if not file.name.startswith('.')))

    if args['first']:
        files = [files[0]]

    for f, file in enumerate(files):
        print(f"Loading {file.name}")
        reshaped = fov_read(path_raw / file.name)
        profile, projected = sharp_planes(reshaped, reference_channel=1, threshold=30)

        if args['color']:
            projected_rgb = channels_combine(projected)

            tf.imwrite(path_rgb / file.name, projected_rgb)

        tf.imwrite(path_projections / file.name, projected)
        if (path_projections / file.name).exists():
            print(f"OK ({file.name})")


if __name__ == '__main__':
    args_test = args_parse()
    main(args_test)
