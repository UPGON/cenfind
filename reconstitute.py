from pathlib import Path

import numpy as np
import tifffile as tf
import cv2


def main():
    path_home = Path('/Volumes/work/datasets')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'

    channels = 'DAPI GFP RFP Cy5 '.split()

    for i in range(5):
        for j in range(5):
            image = np.zeros((4, 67, 2048, 2048), dtype=np.uint16)
            for z in range(67):
                for c, channel in enumerate(channels):
                    plane = tf.imread(
                        path_home / dataset_name / 'by_channel' / f'3-Pos_{i:03}_{j:03}' / f'img_000000000_{c + 1:02} - {channel}_{z:03}.tif')
                    image[c, z, :, :] = plane
            # tf.imwrite(str(path_home / dataset_name / 'raw' / f'{dataset_name}_{i:03}_{j:03}.ome.tif'), image)
            tf.imwrite(str(path_home / dataset_name / 'projected' / f'{dataset_name}_{i:03}_{j:03}.ome.tif'), image.max(1))
            print(image.shape)
            print(f'OK {dataset_name}_{i:03}_{j:03}.tif')


if __name__ == '__main__':
    main()
