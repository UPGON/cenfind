from pathlib import Path
from itertools import product

import cv2
import tifffile as tf
import labelbox

from utils import image_8bit_contrast


def main():
    path_home = Path('/Volumes/work/datasets')
    dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'
    pos_r, pos_c = 0, 0
    channel_id = 2
    path_tiles = path_home / dataset_name / 'tiles'
    path_tiles.mkdir(exist_ok=True)

    data = tf.imread(path_home / dataset_name / 'projections' / f'RPE1wt_CEP63+CETN2+PCNT_1_{pos_r:03}_{pos_c:03}_max.ome.tif', key=channel_id)
    tile_width = 512
    strides = range(0, 2048, tile_width)

    with open('../api_key.txt', 'r') as api_key:
        LB_API_KEY = api_key.read()

    lb = labelbox.Client(api_key=LB_API_KEY)
    dataset = lb.create_dataset(name=f'{dataset_name}_{pos_r:02}_{pos_c:02}_tiles')

    for sx, sy in product(strides, strides):
        print(sx, sy)
        tile = data[sx:sx+tile_width, sy:sy+tile_width]
        tile = image_8bit_contrast(tile)
        tile = cv2.bitwise_not(tile)
        name_dest = str(path_home / dataset_name / 'tiles' / f'tile_{pos_r:02}_{pos_c:02}_{channel_id}_{sx:04}_{sy:04}.png')
        cv2.imwrite(name_dest, tile)
        dataset.create_data_row(row_data=name_dest)


if __name__ == '__main__':
    main()