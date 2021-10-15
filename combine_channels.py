from pathlib import Path
import logging
import tifffile as tf
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def main():
    dyes = {
        (1, 'DAPI'): 'DAPI',
        (2, 'GFP'): 'CEP63',
        (3, 'RFP'): 'CETN2',
        (4, 'Cy5'): 'PCNT',
        (6, 'DIC'): 'DIC',
    }

    path_root = Path('/Volumes/work/epfl/datasets/')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'

    path_raw = path_root / dataset_name / 'raw'
    path_raw.mkdir(exist_ok=True)

    path_channel = path_root / dataset_name / 'by_channel'

    fields = path_channel.iterdir()

    # /Volumes/work/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1/by_channel/3-Pos_004_002/img_000000000_01 - DAPI_000.tif

    for field in fields:
        _, r, c = field.name.split('_')
        shape = (1, 5, 67, 2048, 2048)
        stack = np.zeros(shape, dtype=np.uint16)
        channel_inc = 0
        for (channel_id, channel), marker in dyes.items():
            for plane_id in range(67):
                logging.info(f'Processing {field} {channel} {plane_id}')
                data = tf.imread(
                    path_channel / field.name / f'img_000000000_{channel_id:02} - {channel}_{plane_id:03}.tif')
                stack[:, channel_inc, plane_id, :, :] = data
            channel_inc += 1
        path_field_dst = path_raw / f"{dataset_name}_{r}_{c}.ome.tif"
        tf.imwrite(path_field_dst, stack, photometric='minisblack', shape=shape, metadata={'axes': 'TCZYX'})

        logging.info(f'Saved: {path_field_dst}')



if __name__ == '__main__':
    main()
