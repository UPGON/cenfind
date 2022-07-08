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
import tifffile as tf
from skimage import exposure


def optimise_contrast(data):
    """Apply log contrast adjustment and rescale to uint8"""
    res = exposure.adjust_log(data, gain=3)
    res = exposure.rescale_intensity(res, out_range='uint8')
    return res


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)

    args = parser.parse_args()
    dataset = DataSet(args.path)
    dataset.vignettes.mkdir(exist_ok=True)

    for path in dataset.projections.iterdir():
        projection = tf.imread(path)
        for ch in range(4):
            channel = projection[ch, :, :]
            contrasted = optimise_contrast(channel)
            vignette_name = path.name.replace('_max.tif', f'_max_C{ch}.png')
            cv2.imwrite(str(dataset.vignettes / vignette_name), cv2.bitwise_not(contrasted))


if __name__ == '__main__':
    cli()
