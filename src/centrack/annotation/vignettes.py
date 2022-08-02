import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import exposure

from centrack.layout.dataset import DataSet, FieldOfView


def color_channel(data, color=(1, 1, 0)):
    """
    Create a colored version of a channel image
    :param data:
    :param color:
    :return:
    """
    b = np.multiply(data, color[0], casting='unsafe')
    g = np.multiply(data, color[1], casting='unsafe')
    r = np.multiply(data, color[2], casting='unsafe')
    res = cv2.merge([b, g, r])
    return res


def prepare_channel(fov: FieldOfView, index: int, color: Tuple[int, int, int]):
    res = fov[index]
    res = exposure.rescale_intensity(res, out_range='uint8')
    res = color_channel(res, color)
    return res


def create_vignettes(projection, marker_index: int, nuclei_index: int):
    nuclei = prepare_channel(projection, nuclei_index, (1, 1, 1))
    marker = prepare_channel(projection, marker_index, (0, 1, 0))
    blended = cv2.addWeighted(marker, 0.8, nuclei, 0.2, 50)
    return blended


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('nuclei_index', type=int)
    args = parser.parse_args()

    dataset = DataSet(args.path)
    dataset.vignettes.mkdir(exist_ok=True)

    for path in dataset.projections.iterdir():
        if path.name.startswith('.'):
            continue

        print(f'Processing {path.name}')
        projection = FieldOfView(path)

        for ch in range(4):
            vignette = create_vignettes(projection, ch, args.nuclei_index)
            vignette_name = path.name.replace('.tif', f'_max_C{ch}.png')
            cv2.imwrite(str(dataset.vignettes / vignette_name), vignette)


if __name__ == '__main__':
    cli()
