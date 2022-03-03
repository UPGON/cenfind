import logging
from pathlib import Path

import numpy as np
import tifffile as tf
from cv2 import cv2
from skimage import exposure

from src.centrack.commands.status import Marker
from src.centrack.commands import contrast

logging.basicConfig(level=logging.DEBUG)


def improve(image, percentile=(.1, 99.9)):
    percentiles = np.percentile(image, percentile)
    return exposure.rescale_intensity(image, in_range=tuple(percentiles), out_range='uint8')


projection = '_max'

mm_suffix = '_MMStack_Default'

dataset_name = '20210727_HA-FL-SAS6_Clones'

filename = '20210728_HA-FL_S6_CloneC2_DAPI+rPOC5AF488+mHA568+gCPAP647_R1_4_MMStack_Default_max'
filename_stripped = filename.replace(projection, '').replace(mm_suffix, '')

metadata_keys = ['date', 'genotype', 's', 'clone', 'markers', 'replicate']

metadata_dict = {key: value for key, value in zip(metadata_keys, filename.split('_'))}

condition = metadata_dict['genotype']
markers_list = metadata_dict['markers'].split('+')
markers = [Marker.from_code(code, position=position)
           for position, code in enumerate(markers_list)]
replicate = metadata_dict['replicate']

# condition, markers, replicate = dataset_name.split('_')
# markers_list = markers.split('+')
# markers_list.insert(0, 'DAPI')

path_root = Path('/Volumes/work/epfl/datasets')
path_dataset = path_root / dataset_name
path_projections = path_dataset / 'projections'

if __name__ == '__main__':

    for field in path_projections.rglob('**/*.tif'):
        if field.name.startswith('.'):
            continue

        name_core = field.name.rstrip('.tif')
        data = tf.imread(field)

        for c, marker in enumerate(markers):
            channel = marker.protein
            path_projections_channel = path_dataset / 'projections_channel' / channel

            path_projections_channel.mkdir(exist_ok=True, parents=True)
            (path_projections_channel / 'tif').mkdir(exist_ok=True)
            (path_projections_channel / 'png').mkdir(exist_ok=True)

            file_name_dst = f"{name_core}_C{c}"

            data_yxc = np.moveaxis(data, 0, -1)
            plane = data_yxc[:, :, c]
            tf.imwrite(path_projections_channel / 'tif' / (file_name_dst + ".tif"), plane)

            res = cv2.bitwise_not(contrast(plane))
            cv2.putText(res, f"{file_name_dst} ({channel})", (200, 200), cv2.QT_FONT_NORMAL, .8,
                        color=(255, 255, 255),
                        thickness=2)

            cv2.imwrite(str(path_projections_channel / 'png' / (file_name_dst + ".png")), res)
            logging.info(f'Saved: {name_core} channel: {c}')
