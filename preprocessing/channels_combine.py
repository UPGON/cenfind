import cv2
import tifffile as tf
from pathlib import Path
import numpy as np

from utils import image_8bit_contrast, channels_combine


def main():
    path_root = Path('/Volumes/work/datasets/RPE1wt_CEP63+Cyclin-PCNA+PCNT_1')

    path_projections = path_root / 'projected'
    path_projections.mkdir(exist_ok=True)

    path_rgb = path_root / 'color'
    path_rgb.mkdir(exist_ok=True)

    files = sorted(tuple(file for file in path_projections.iterdir()
                         if file.name.endswith('.tif')
                         if not file.name.startswith('.')))

    for file in files:
        print(f"Loading {file}")

        # noinspection PyTypeChecker
        projected = tf.imread(file, key=range(4))
        file_name_rgb = file.name.split('.')[0] + '.png'
        print(f'Saving {str(path_rgb / file_name_rgb)}')
        final = channels_combine(projected)
        cv2.imwrite(str(path_rgb / file_name_rgb), final)


if __name__ == '__main__':
    main()
