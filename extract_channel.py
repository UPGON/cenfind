from pathlib import Path

import tifffile as tf
import cv2
from aicsimageio import AICSImage

from centrack.data import contrast

dataset_name = 'RPE1wt_CEP152+GTU88+PCNT_1'
# channel_id = [1, 2, 3]

condition, markers, replicate = dataset_name.split('_')
markers_list = markers.split('+')
markers_list.insert(0, 'DAPI')
# marker = markers_list[channel_id]

path_root = Path('/Volumes/work/epfl/datasets')
path_dataset = path_root / dataset_name
path_projections = path_dataset / 'projections'


if __name__ == '__main__':

    for field in path_projections.rglob('**/*.ome.tif'):
        if field.name.startswith('.'):
            continue

        name_core = field.name.rstrip('.ome.tif')
        data = AICSImage(field)

        for c in range(data.dims.C):
            channel = markers_list[c]
            print(f'Process {field.name} {channel}.')
            plane = data.get_image_data('YX', C=c)

            path_projections_channel = path_dataset / 'projections_channel' / channel
            path_projections_channel.mkdir(exist_ok=True, parents=True)

            (path_projections_channel / 'tif').mkdir(exist_ok=True)
            (path_projections_channel / 'png').mkdir(exist_ok=True)
            tf.imwrite(path_projections_channel / 'tif' / f"{name_core}_C{c}.tif", plane)

            contrasted = contrast(plane)
            inverted = cv2.bitwise_not(contrasted)
            cv2.imwrite(str(path_projections_channel / 'png' / f"{name_core}_C{c}.png"), inverted)
            print(f'Saved: {name_core} channel: {c}')