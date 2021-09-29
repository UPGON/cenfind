from operator import itemgetter
from pathlib import Path
from datetime import datetime as dt

import cv2
import numpy as np
import tifffile as tf
import pytomlpp

from aicsimageio import AICSImage

from centrack.data import Plane, contrast
from centrack.annotation import color_scale
from centrack.detectors import FocusDetector

# Foci
RED_BGR = (44, 36, 187)
RED_BGR_SCALED = color_scale(RED_BGR)

# Foci label
VIOLET_BGR = (251, 112, 154)
VIOLET_BGR_SCALED = color_scale(VIOLET_BGR)

# Nuclei
BLUE_BGR = (251, 139, 94)
BLUE_BGR_SCALED = color_scale(BLUE_BGR)

# Annotation
WHITE = (255, 255, 255)


def main():
    config = pytomlpp.load('../configs/config.toml')

    config_dataset = config['dataset']
    path_root = Path(config_dataset['root'])
    dataset_name = config_dataset['name']

    config_data = config['data']
    channel_id = config_data['channel']
    x, y = config_data['position']

    # DATA LOADING
    fov_core = f'{dataset_name}_{x:03}_{y:03}'
    fov_name = fov_core + '_max.ome.tif'
    path_projected = path_root / f'{dataset_name}' / 'projections' / fov_name
    path_out = path_root / dataset_name / 'out'
    path_out.mkdir(exist_ok=True)

    field = AICSImage(path_projected)
    data = field.get_image_data('ZYX', C=channel_id).max(axis=0).squeeze()
    fov = Plane(data, field)

    # SEMANTIC ANNOTATION
    foci_mask = (fov
                 .blur_median(3)
                 .maximum_filter(size=5)
                 .contrast()
                 .threshold(threshold=20)
                 )

    cv2.imwrite(str(path_out / f"{fov_core}_mask.png"), foci_mask.data)

    foci_detector = FocusDetector(fov, organelle='centriole')
    rois = foci_detector.detect()

    annotation_map = cv2.cvtColor(fov.data, cv2.COLOR_GRAY2BGR)
    annotation_map = contrast(annotation_map)

    path_crop = path_root / dataset_name / 'crops'
    path_crop.mkdir(exist_ok=True)

    for roi in rois:
        roi.draw(annotation_map, color=RED_BGR)

    for roi in rois:
        crop = roi.extract(annotation_map)
        roi_id = roi.idx
        cv2.imwrite(str(path_crop / f'{fov_name.split(".")[0]}_C{channel_id}_{roi_id}.png'), crop)

    cv2.imwrite(str(path_out / f"{fov_core}_annot_foci.png"), annotation_map)

    # cv2.imwrite(str(path_out / f'{fov_name.split(".")[0]}_C{channel_id}_iou.png'), comparison)
    #
    # print(f'Threshold: ? Foci detected: {len(foci_coords)} IoU: {iou:>.03}')
    #
    # config['metrics'] = {}
    # config['metrics']['foci_detected'] = len(foci_detector)
    # config['metrics']['iou'] = iou
    # timestamp = dt.now()
    # pytomlpp.dump(config, path_root / dataset_name / f'config_out_{timestamp}.toml')


if __name__ == '__main__':
    main()
