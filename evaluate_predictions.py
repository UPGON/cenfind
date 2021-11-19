import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

from centrack.data import contrast
from centrack.metrics import generate_synthetic_data, generate_predictions, compute_metrics

logging.basicConfig(level=logging.INFO)

CYAN_RGB = (0, 255, 255)
RED_RGB = (255, 0, 0)
WHITE_RGB = (255, 255, 255)

path_root = Path('/Users/leoburgy/Dropbox/epfl/centriole_detection')
path_data = path_root / 'data'
path_test = path_data / 'test'
path_annotation = path_data / 'annotations.csv'
path_pred_annotation = path_root / 'predictions/annotations'
path_pred_vis = path_root / 'predictions/visualisation'
path_pred_vis.mkdir(exist_ok=True)


def metric_for_image(image_name):
    path_file = path_test / image_name
    image_stem = path_file.stem

    if not path_file:
        height = 128
        size = 10
        positions = generate_synthetic_data(height=height, size=size)
        predictions = generate_predictions(height=height, foci=positions,
                                           fp_rate=.1, fn_rate=.05)
        image = np.zeros((height, height, 3), dtype=np.uint8)

        for focus in positions:
            r, c = focus
            cv2.circle(image, radius=2, center=(r, c), color=WHITE_RGB, thickness=-1)

        image = cv2.GaussianBlur(image, (3, 3), 0)

    else:
        annot_actual = pd.read_csv(path_annotation)
        positions = annot_actual.loc[annot_actual['image_name'] == image_stem]
        predictions = pd.read_csv(path_pred_annotation / (image_stem + '.tif.csv'))

        positions = positions[['x', 'y']].to_numpy()
        predictions = predictions[['x', 'y']].to_numpy()

        image = tf.imread(path_test / (image_stem + '.tif'))
        image = contrast(image)

    results = compute_metrics(positions, predictions, offset_max=4)

    fps = results['fp']
    fns = results['fn']
    tps = results['tp']

    height, width = image.shape
    annotation = np.zeros((height, width, 3), dtype=np.uint8)

    for idx in fps:
        r, c = predictions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=RED_RGB, thickness=2,
                       markerSize=int(10 / np.sqrt(2)), markerType=cv2.MARKER_TILTED_CROSS)

    for idx in fns:
        r, c = positions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=CYAN_RGB, thickness=1,
                       markerSize=10, markerType=cv2.MARKER_SQUARE)

    for idx in tps:
        r, c = predictions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=CYAN_RGB, thickness=2,
                       markerSize=10, markerType=cv2.MARKER_CROSS)

    # Write the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(image_rgb, .8, annotation, .5, 0)
    tf.imwrite(path_pred_vis / (image_stem + '_annot.png'), result)

    metrics = {
        'fp': len(fps),
        'fn': len(fns),
        'tp': len(tps)
    }

    logging.info(metrics)

    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'])
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'])

    logging.info('precision : %.3f', precision)
    logging.info('recall : %.3f', recall)


def main():
    for image in path_test.iterdir():
        if image.name.startswith('.') or not image.name.endswith('tif'):
            continue
        logging.info(image.name)
        metric_for_image(image)


if __name__ == '__main__':
    main()
