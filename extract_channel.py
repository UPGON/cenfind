from pathlib import Path

import tifffile as tf
import cv2
from aicsimageio import AICSImage

from centrack.data import contrast

dataset_name = '20210727_HA-FL-SAS6_Clones'
channel_id = 1

condition, markers, replicate = dataset_name.split('_')
markers_list = markers.split('+')
markers_list.insert(0, 'DAPI')
marker = markers_list[channel_id]

path_root = Path('/Volumes/work/datasets')
path_dataset = path_root / dataset_name
path_projections = path_dataset / 'projections'

path_projections_channel = path_dataset / 'projections_channel' / marker
path_projections_channel.mkdir(exist_ok=True, parents=True)

(path_projections_channel / 'tif').mkdir(exist_ok=True)
(path_projections_channel / 'png').mkdir(exist_ok=True)


if __name__ == '__main__':

    for field in path_projections.rglob('**/*.ome.tif'):
        if field.name.startswith('.'):
            continue
        name_core = field.name.rstrip('.ome.tif')
        data = AICSImage(field)
        plane = data.get_image_data('YX', C=channel_id)
        tf.imwrite(path_projections_channel / 'tif' / f"{name_core}_C{channel_id}.ome.tif", plane)

        contrasted = contrast(plane)
        cv2.imwrite(str(path_projections_channel / 'png' / f"{name_core}_C{channel_id}.png"), contrasted)
        print(f'Saved: {name_core}')