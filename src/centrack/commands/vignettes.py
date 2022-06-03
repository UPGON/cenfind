"""
Create vignettes for projections
- png
- 8bit
- by channel, like projections
- contrast adjusted CLAHE
"""
import argparse
from pathlib import Path
from centrack.commands.status import DataSet
from cv2 import cv2
import tifffile as tf
from skimage import exposure


def optimise_contrast(data):
    """Apply log contrast adjustment and rescale to uint8"""
    res = exposure.adjust_log(data, gain=3)
    res = exposure.rescale_intensity(res, out_range='uint8')
    return res


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('channel', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    opts = args_parse()
    dataset = DataSet(Path(opts.path))
    dataset.vignettes.mkdir(exist_ok=True)
    data_rows = []

    for projection in dataset.projections.iterdir():
        if projection.stem.endswith(f'C{opts.channel}'):
            channel = tf.imread(projection)
            external_id = projection.stem
            contrasted = optimise_contrast(channel)
            cv2.imwrite(str(dataset.vignettes / f"{projection.stem}.png"), cv2.bitwise_not(contrasted))
