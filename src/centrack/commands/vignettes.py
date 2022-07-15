"""
Create vignettes for projections
- png
- 8bit
- by channel, like projections
"""
import argparse
from pathlib import Path
from centrack.utils.status import DataSet
import cv2
import numpy as np
import tifffile as tf
from skimage import exposure


def optimise_contrast(data):
    """Apply log contrast adjustment and rescale to uint8"""
    res = exposure.adjust_log(data, gain=3)
    res = exposure.rescale_intensity(res, out_range='uint8')
    return res


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


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)

    args = parser.parse_args()
    dataset = DataSet(args.path)
    dataset.vignettes.mkdir(exist_ok=True)

    for path in dataset.projections.iterdir():
        projection = tf.imread(path)
        nuclei = optimise_contrast(projection[0, :, :])
        nuclei_bgr = cv2.cvtColor(nuclei, cv2.COLOR_GRAY2BGR)
        for ch in range(4):
            print(f'Processing {path.name} {ch}')
            channel = projection[ch, :, :]
            channel = optimise_contrast(channel)
            channel_bgr = color_channel(channel, color=(0, 1, 0))
            blended = cv2.addWeighted(channel_bgr, 0.8, nuclei_bgr, 0.2, 100)
            vignette_name = path.name.replace('_max.tif', f'_max_C{ch}.png')
            cv2.imwrite(str(dataset.vignettes / vignette_name), blended)


if __name__ == '__main__':
    cli()
