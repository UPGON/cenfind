import logging
from pathlib import Path

import tifffile as tf
import numpy as np

logging.basicConfig(level=logging.DEBUG)

dyes = {
    (1, 'DAPI'): 'DAPI',
    (2, 'GFP'): 'CEP63',
    (3, 'RFP'): 'CETN2',
    (4, 'Cy5'): 'PCNT',
    (6, 'DIC'): 'DIC',
}


def field_combine(path_field_src, shape, dyes):
    """Accumulate planes of a field of view."""
    t, c, z, y, x = shape
    stack = np.zeros(shape, dtype=np.uint16)
    channel_inc = 0
    for (channel_id, channel), marker in dyes.items():
        for plane_id in range(z):
            logging.info(f'Processing {path_field_src} {channel} {plane_id}')
            data = tf.imread(
                path_field_src.name / f'img_000000000_{channel_id:02} - {channel}_{plane_id:03}.tif')
            stack[:, channel_inc, plane_id, :, :] = data
        channel_inc += 1
    return stack


def main():
    path_root = Path('/Volumes/work/epfl/datasets/')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'

    path_raw = path_root / dataset_name / 'raw'
    path_raw.mkdir(exist_ok=True)

    path_channel = path_root / dataset_name / 'by_channel'

    fields = path_channel.iterdir()

    for field in fields:
        _, r, c = field.name.split('_')
        path_field_dst = path_raw / f"{dataset_name}_{r}_{c}.ome.tif"

        shape = (1, 5, 67, 2048, 2048)
        stack = field_combine(field, dyes, shape)

        tf.imwrite(path_field_dst, stack, photometric='minisblack', shape=shape, metadata={'axes': 'TCZYX'})
        logging.info(f'Saved: {path_field_dst}')


if __name__ == '__main__':
    main()
