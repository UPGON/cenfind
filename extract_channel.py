from pathlib import Path
import logging

import tifffile as tf
import cv2

from centrack.data import contrast

logging.basicConfig(level=logging.DEBUG)

dataset_name = 'RPE1wt_CEP152+GTU88+PCNT_1'

condition, markers, replicate = dataset_name.split('_')
markers_list = markers.split('+')
markers_list.insert(0, 'DAPI')

path_root = Path('/Volumes/work/epfl/datasets')
path_dataset = path_root / dataset_name
path_projections = path_dataset / 'projections'

if __name__ == '__main__':

    for field in path_projections.rglob('**/*.ome.tif'):
        if field.name.startswith('.'):
            continue

        name_core = field.name.rstrip('.ome.tif')
        data = tf.imread(field)

        for c, channel in enumerate(markers_list):
            path_projections_channel = path_dataset / 'projections_channel' / channel

            path_projections_channel.mkdir(exist_ok=True, parents=True)
            (path_projections_channel / 'tif').mkdir(exist_ok=True)
            (path_projections_channel / 'png').mkdir(exist_ok=True)

            file_name_dst = f"{name_core}_C{c}"

            plane = data[c, :, :]
            tf.imwrite(path_projections_channel / 'tif' / (file_name_dst + ".tif"), plane)

            contrasted = contrast(plane)
            inverted = cv2.bitwise_not(contrasted)
            cv2.putText(inverted, f"{file_name_dst} ({channel})", (200, 200), cv2.QT_FONT_NORMAL, .8, color=10, thickness=2)
            cv2.imwrite(str(path_projections_channel / 'png' / (file_name_dst + ".png")), inverted)
            logging.info(f'Saved: {name_core} channel: {c}')
