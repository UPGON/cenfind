from pathlib import Path

import cv2

from centrack.utils import channels_combine
import tifffile as tf


def main():
    path_dataset = Path('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1')
    path_projections = path_dataset / 'projections'
    path_color = (path_dataset / 'color')
    path_color.mkdir(exist_ok=True)

    for path in path_projections.iterdir():
        if path.name.endswith('tif') and not path.name.startswith('.'):
            data = tf.imread(path)
            color = channels_combine(data, (1, 2, 3))
            color = cv2.bitwise_not(color)
            stem = path.name.removesuffix(''.join(path.suffixes))
            tf.imwrite(path_color / f'{stem}_color.png', color)


if __name__ == '__main__':
    main()
