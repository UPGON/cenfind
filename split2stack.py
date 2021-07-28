from pathlib import Path

import numpy as np
import tifffile as tf


def main():
    path_home = Path('/Volumes/work/datasets')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'

    channels = 'DAPI GFP RFP Cy5'.split()

    for i in range(5):
        for j in range(5):
            image = np.zeros((67, 4, 2048, 2048), dtype=np.uint16)
            for z in range(67):
                for c, channel in enumerate(channels):
                    print(i, j, z, c)
                    plane = tf.imread(
                        path_home / dataset_name / 'by_channel' / f'3-Pos_{i:03}_{j:03}' / f'img_000000000_{c + 1:02} - {channel}_{z:03}.tif')
                    image[z, c, :, :] = plane

            path_dest = Path(path_home / dataset_name / 'raw' / f'{dataset_name}_{i:03}_{j:03}.ome.tif')
            with tf.TiffWriter(path_dest) as tif:
                    tif.write(image, photometric='minisblack')

            print(f'OK {dataset_name}_{i:03}_{j:03}.tif')


if __name__ == '__main__':
    main()
